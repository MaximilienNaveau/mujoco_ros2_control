/**
 * Copyright (c) 2026, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 *
 * All rights reserved.
 *
 * This software is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "camera_plugin.hpp"

namespace mujoco_ros2_control_plugins
{

bool CameraPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data)
{
  // by default use `glfwInit` to check if the GLFW is initialized.
  return this->init(node, model, data, glfwInit);
}

bool CameraPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data, GlfwInitFn glfw_init_fn)
{
  node_ = node;
  mj_model_ = model;
  mj_data_ = data;

  // Ensure the logger has a name
  logger_ = node_->get_logger().get_child(node->get_sub_namespace());
  RCLCPP_INFO(node_->get_logger(), "CameraPlugin initializing cameras...");

  // Read the mj_model_, identify the number of cameras, and populate containers for them.
  if (!register_cameras())
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to register cameras.");
    return false;
  }
  if (cameras_.empty())
  {
    return true;
  }

  // Start the rendering thread process
  if (render_backend_ == "egl")
  {
    // Asked for explicitly, so do not touch GLFW at all: the point of choosing EGL on a
    // machine that has a display is to stay off the X queue the viewer draws on.
    RCLCPP_INFO(node_->get_logger(), "Using EGL for camera rendering (render_backend=egl).");
    use_egl_ = true;
  }
  else if (render_backend_ == "glfw")
  {
    if (!glfw_init_fn())
    {
      RCLCPP_ERROR(node_->get_logger(), "render_backend=glfw was requested but GLFW failed to initialize.");
      return false;
    }
    use_egl_ = false;
  }
  else
  {
    // Try GLFW first, fall back to EGL for headless environments
    if (glfw_init_fn())
    {
      use_egl_ = false;
    }
    else
    {
      RCLCPP_WARN(node_->get_logger(), "Failed to initialize GLFW. Attempting EGL for headless rendering.");
      use_egl_ = true;
    }
  }
  rendering_thread_ = std::thread(&CameraPlugin::update_loop, this);
  return true;
}

void CameraPlugin::update(const mjModel* model_arg, mjData* data)
{
  if (!publish_images_)
  {
    return;
  }

  // Streaming cameras render on a fixed-rate clock; polled cameras render as soon as a
  // trigger has been received. Both are serviced here, on the sim thread, because this is
  // the only place we can safely snapshot the live mjData without racing the simulation.
  // TODO: Support per-camera publish rates?
  const auto now = node_->get_clock()->now();
  const bool stream_due =
      has_streaming_cameras_ && (now - last_publish_time_).seconds() >= (1.0 / camera_publish_rate_);
  const bool poll_due = poll_pending_.exchange(false);

  // Nothing to do this step: avoid taking the lock or copying data.
  if (!stream_due && !poll_due)
  {
    return;
  }

  bool any_selected = false;
  {
    std::unique_lock<std::mutex> lock(data_mutex_);

    // The rendering thread is still working on the previous snapshot. Touching
    // mj_camera_data_ or render_pending now would corrupt the frame it is rendering, so
    // give up this slot instead. See CameraPlugin::render_in_flight_.
    if (render_in_flight_)
    {
      // A poll trigger must survive the skip. poll_pending_ was already consumed by the
      // exchange above, but the per-camera poll_requested flag is still set, so re-arm the
      // wakeup hint and the request is merely deferred rather than lost.
      if (poll_due)
      {
        poll_pending_.store(true);
      }
      if (stream_due)
      {
        // Consume the slot so the next attempt is a full interval away. Without this, every
        // control cycle for the rest of the render would re-enter here and the counter
        // would report attempts (thousands per second at a 2 kHz control rate) rather than
        // lost frames.
        last_publish_time_ = now;
        ++dropped_frames_;
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                             "Camera rendering cannot keep up: %llu streaming frame(s) dropped so far. "
                             "Reduce mujoco_plugins.mujoco_camera_plugin.camera_publish_rate, the camera "
                             "resolution, or the number of streaming cameras; or lower sim_speed_factor, which "
                             "keeps the sim-time image rate and gives each pass more wall-clock time.",
                             static_cast<unsigned long long>(dropped_frames_.load()));
      }
      return;
    }

    // Flag streaming cameras when their interval is due and any polled cameras that have a
    // pending trigger (consuming the one-shot request so they render exactly once).
    for (auto& camera : cameras_)
    {
      if (camera.policy == CameraPolicy::STREAMING && stream_due)
      {
        camera.render_pending = true;
        any_selected = true;
      }
      else if (camera.policy == CameraPolicy::POLLED && camera.poll_requested)
      {
        camera.poll_requested = false;
        camera.render_pending = true;
        any_selected = true;
      }
    }

    if (stream_due)
    {
      last_publish_time_ = now;
    }

    // Only snapshot the simulation data when there is actually a camera to render.
    if (any_selected)
    {
      mjv_copyData(mj_camera_data_, model_arg, data);
      new_data_ = true;
      // Hand ownership of the snapshot to the rendering thread; it is released again once
      // update_cameras() has published. Set under the lock so the flag can never be
      // observed as false while new_data_ is already true.
      render_in_flight_ = true;
    }
  }

  if (any_selected)
  {
    data_cv_.notify_one();
  }
}

void CameraPlugin::cleanup()
{
  close();
}

void CameraPlugin::trigger_update()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  // Force every streaming camera and consume any pending polls, then render synchronously.
  for (auto& camera : cameras_)
  {
    if (camera.policy == CameraPolicy::STREAMING)
    {
      camera.render_pending = true;
    }
    else if (camera.policy == CameraPolicy::POLLED && camera.poll_requested)
    {
      camera.poll_requested = false;
      camera.render_pending = true;
    }
  }
  update_cameras();
}

bool CameraPlugin::register_cameras()
{
  const std::string param_prefix = "mujoco_plugins." + node_->get_sub_namespace() + ".";
  const std::string camera_publish_rate_param = param_prefix + "camera_publish_rate";
  if (!node_->has_parameter(camera_publish_rate_param))
  {
    node_->declare_parameter(camera_publish_rate_param, 5.0);
  }

  camera_publish_rate_ = node_->get_parameter(camera_publish_rate_param).as_double();
  RCLCPP_INFO(node_->get_logger(), "Publishing camera data at rate %f per second.", camera_publish_rate_);

  // Depth sensor model. MuJoCo returns exact geometric depth at every pixel: no noise,
  // no invalid returns, and values below the real Min-Z. A map or an estimator built
  // against that looks far better in simulation than it can on hardware, so the sensor
  // is modelled here rather than letting the consumer discover the difference on the
  // robot.
  if (!node_->has_parameter(param_prefix + "depth_sensor_model"))
  {
    node_->declare_parameter(param_prefix + "depth_sensor_model", true);
  }
  depth_sensor_model_ = node_->get_parameter(param_prefix + "depth_sensor_model").as_bool();
  if (!node_->has_parameter(param_prefix + "depth_min_range"))
  {
    node_->declare_parameter(param_prefix + "depth_min_range", 0.28);
  }
  depth_min_range_ = static_cast<float>(
      node_->get_parameter(param_prefix + "depth_min_range").as_double());
  if (!node_->has_parameter(param_prefix + "depth_max_range"))
  {
    node_->declare_parameter(param_prefix + "depth_max_range", 3.0);
  }
  depth_max_range_ = static_cast<float>(
      node_->get_parameter(param_prefix + "depth_max_range").as_double());
  if (!node_->has_parameter(param_prefix + "depth_stereo_baseline"))
  {
    node_->declare_parameter(param_prefix + "depth_stereo_baseline", 0.05);
  }
  depth_stereo_baseline_ = static_cast<float>(
      node_->get_parameter(param_prefix + "depth_stereo_baseline").as_double());
  if (!node_->has_parameter(param_prefix + "depth_subpixel_error"))
  {
    node_->declare_parameter(param_prefix + "depth_subpixel_error", 0.15);
  }
  depth_subpixel_error_ = static_cast<float>(
      node_->get_parameter(param_prefix + "depth_subpixel_error").as_double());
  if (!node_->has_parameter(param_prefix + "depth_dropout_fraction"))
  {
    node_->declare_parameter(param_prefix + "depth_dropout_fraction", 0.02);
  }
  depth_dropout_fraction_ = node_->get_parameter(param_prefix + "depth_dropout_fraction").as_double();
  if (depth_sensor_model_)
  {
    RCLCPP_INFO(node_->get_logger(),
                "Depth sensor model ON: range [%.2f, %.2f] m, stereo baseline %.3f m, "
                "subpixel error %.3f px, dropout %.1f%%. Invalid pixels are NaN.",
                depth_min_range_, depth_max_range_, depth_stereo_baseline_,
                depth_subpixel_error_, 100.0 * depth_dropout_fraction_);
  }

  if (!node_->has_parameter(param_prefix + "render_backend"))
  {
    node_->declare_parameter(param_prefix + "render_backend", "auto");
  }
  render_backend_ = node_->get_parameter(param_prefix + "render_backend").as_string();
  if (render_backend_ != "auto" && render_backend_ != "glfw" && render_backend_ != "egl")
  {
    RCLCPP_ERROR(node_->get_logger(),
                 "Unknown render_backend '%s'; expected \"auto\", \"glfw\" or \"egl\".",
                 render_backend_.c_str());
    return false;
  }

  cameras_.resize(0);
  for (auto i = 0; i < mj_model_->ncam; ++i)
  {
    const char* cam_name = mj_model_->names + mj_model_->name_camadr[i];
    const int* cam_resolution = mj_model_->cam_resolution + 2 * i;
    const mjtNum cam_fovy = mj_model_->cam_fovy[i];

    // Construct CameraData wrapper and set defaults
    CameraData camera;
    camera.name = cam_name;
    camera.mjv_cam.type = mjCAMERA_FIXED;
    camera.mjv_cam.fixedcamid = i;
    camera.width = static_cast<uint32_t>(cam_resolution[0]);
    camera.height = static_cast<uint32_t>(cam_resolution[1]);
    camera.viewport = { 0, 0, cam_resolution[0], cam_resolution[1] };

    const std::string param_ns = param_prefix + cam_name + ".";

    const std::string policy_param = param_ns + "policy";
    if (!node_->has_parameter(policy_param))
    {
      node_->declare_parameter(policy_param, "streaming");
    }
    std::string policy_str = node_->get_parameter(policy_param).as_string();
    if (policy_str == "streaming")
    {
      camera.policy = CameraPolicy::STREAMING;
    }
    else if (policy_str == "polled")
    {
      camera.policy = CameraPolicy::POLLED;
    }
    else if (policy_str == "disabled")
    {
      camera.policy = CameraPolicy::DISABLED;
    }
    else
    {
      RCLCPP_ERROR(node_->get_logger(), "Invalid policy for camera '%s'", cam_name);
      return false;
    }

    const std::string frame_param = param_ns + "frame_name";
    if (!node_->has_parameter(frame_param))
    {
      node_->declare_parameter(frame_param, camera.name + "_frame");
    }
    camera.frame_name = node_->get_parameter(frame_param).as_string();

    if (!node_->has_parameter(param_ns + "info_topic"))
    {
      node_->declare_parameter(param_ns + "info_topic", camera.name + "/camera_info");
    }
    camera.info_topic = node_->get_parameter(param_ns + "info_topic").as_string();

    if (!node_->has_parameter(param_ns + "image_topic"))
    {
      node_->declare_parameter(param_ns + "image_topic", camera.name + "/color");
    }
    camera.image_topic = node_->get_parameter(param_ns + "image_topic").as_string();

    if (!node_->has_parameter(param_ns + "depth_topic"))
    {
      node_->declare_parameter(param_ns + "depth_topic", camera.name + "/depth");
    }
    camera.depth_topic = node_->get_parameter(param_ns + "depth_topic").as_string();

    if (!node_->has_parameter(param_ns + "trigger_service_name"))
    {
      node_->declare_parameter(param_ns + "trigger_service_name", camera.name + "/trigger");
    }
    camera.trigger_service_name = node_->get_parameter(param_ns + "trigger_service_name").as_string();

    RCLCPP_INFO(node_->get_logger(), "Adding camera: '%s'", cam_name);
    RCLCPP_INFO(node_->get_logger(), "    policy: '%s'", policy_str.c_str());
    RCLCPP_INFO(node_->get_logger(), "    frame_name: '%s'", camera.frame_name.c_str());
    if (camera.policy == CameraPolicy::DISABLED)
    {
      continue;
    }
    RCLCPP_INFO(node_->get_logger(), "    info_topic: '%s'", camera.info_topic.c_str());
    RCLCPP_INFO(node_->get_logger(), "    image_topic: '%s'", camera.image_topic.c_str());
    RCLCPP_INFO(node_->get_logger(), "    depth_topic: '%s'", camera.depth_topic.c_str());
    if (camera.policy == CameraPolicy::POLLED)
    {
      RCLCPP_INFO(node_->get_logger(), "    trigger_service_name: '%s'", camera.trigger_service_name.c_str());
    }

    // Configure publishers and services
    camera.camera_info_pub = node_->create_publisher<sensor_msgs::msg::CameraInfo>(camera.info_topic, 1);
    camera.image_pub = node_->create_publisher<sensor_msgs::msg::Image>(camera.image_topic, 1);
    camera.depth_image_pub = node_->create_publisher<sensor_msgs::msg::Image>(camera.depth_topic, 1);
    if (camera.policy == CameraPolicy::POLLED)
    {
      camera.trigger_service = node_->create_service<std_srvs::srv::Trigger>(
          camera.trigger_service_name, [this, i](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            this->handle_trigger(request, response, i);
          });
    }
    // Setup containers for color image data
    camera.image.header.frame_id = camera.frame_name;

    const auto image_size = camera.width * camera.height * 3;
    camera.image_buffer.resize(image_size);
    camera.image.data.resize(image_size);
    camera.image.width = camera.width;
    camera.image.height = camera.height;
    camera.image.step = camera.width * 3;
    camera.image.encoding = sensor_msgs::image_encodings::RGB8;

    // Depth image data
    camera.depth_image.header.frame_id = camera.frame_name;
    camera.depth_buffer.resize(camera.width * camera.height);
    camera.depth_image.data.resize(camera.width * camera.height * sizeof(float));
    camera.depth_image.width = camera.width;
    camera.depth_image.height = camera.height;
    camera.depth_image.step = camera.width * sizeof(float);
    camera.depth_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;

    // Camera info
    camera.camera_info.header.frame_id = camera.frame_name;
    camera.camera_info.width = camera.width;
    camera.camera_info.height = camera.height;
    camera.camera_info.distortion_model = "plumb_bob";
    camera.camera_info.k.fill(0.0);
    camera.camera_info.r.fill(0.0);
    camera.camera_info.p.fill(0.0);
    camera.camera_info.d.resize(5, 0.0);

    double focal_scaling = (1.0 / std::tan((cam_fovy * M_PI / 180.0) / 2.0)) * camera.height / 2.0;
    camera.camera_info.k[0] = camera.camera_info.p[0] = focal_scaling;
    camera.camera_info.k[2] = camera.camera_info.p[2] = static_cast<double>(camera.width) / 2.0;
    camera.camera_info.k[4] = camera.camera_info.p[5] = focal_scaling;
    camera.camera_info.k[5] = camera.camera_info.p[6] = static_cast<double>(camera.height) / 2.0;
    camera.camera_info.k[8] = camera.camera_info.p[10] = 1.0;

    // Add to list of cameras
    cameras_.push_back(camera);
  }

  has_streaming_cameras_ = std::any_of(cameras_.begin(), cameras_.end(),
                                       [](const CameraData& cam) { return cam.policy == CameraPolicy::STREAMING; });

  // Reserve once so the per-pass render bookkeeping in `update_cameras()` never allocates.
  render_indices_.reserve(cameras_.size());

  return true;
}

void CameraPlugin::close()
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_requested_ = true;
    publish_images_ = false;
  }
  data_cv_.notify_one();
  if (rendering_thread_.joinable())
  {
    rendering_thread_.join();
  }
}

bool CameraPlugin::init_egl_context()
{
  // Get EGL display
  egl_display_ = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);

  // Initialize EGL
  EGLint major, minor;
  if (!eglInitialize(egl_display_, &major, &minor))
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to initialize (error: 0x%x)", eglGetError());
    return false;
  }
  const char* client_apis = eglQueryString(egl_display_, EGL_CLIENT_APIS);
  const char* vendor = eglQueryString(egl_display_, EGL_VENDOR);
  RCLCPP_INFO(node_->get_logger(), "EGL: Initialized version %d.%d", major, minor);
  RCLCPP_INFO(node_->get_logger(), "EGL: Vendor: %s, APIs: %s", vendor, client_apis);
  // Choose EGL config for offscreen rendering
  const EGLint config_attribs[] = { EGL_SURFACE_TYPE,
                                    EGL_PBUFFER_BIT,
                                    EGL_RED_SIZE,
                                    8,
                                    EGL_GREEN_SIZE,
                                    8,
                                    EGL_BLUE_SIZE,
                                    8,
                                    EGL_ALPHA_SIZE,
                                    8,
                                    EGL_DEPTH_SIZE,
                                    24,
                                    EGL_RENDERABLE_TYPE,
                                    EGL_OPENGL_BIT,
                                    EGL_NONE };

  EGLConfig egl_config;
  EGLint num_configs;
  if (!eglChooseConfig(egl_display_, config_attribs, &egl_config, 1, &num_configs) || num_configs == 0)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to choose config (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Bind OpenGL API
  if (!eglBindAPI(EGL_OPENGL_API))
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to bind OpenGL API (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Create EGL context
  egl_context_ = eglCreateContext(egl_display_, egl_config, EGL_NO_CONTEXT, nullptr);
  if (egl_context_ == EGL_NO_CONTEXT)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to create context (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Create PBuffer surface for offscreen rendering
  const EGLint pbuffer_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
  egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config, pbuffer_attribs);
  if (egl_surface_ == EGL_NO_SURFACE)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to create PBuffer surface (error: 0x%x)", eglGetError());
    eglDestroyContext(egl_display_, egl_context_);
    eglTerminate(egl_display_);
    egl_context_ = EGL_NO_CONTEXT;
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Make the context current.
  //
  // The pbuffer above is requested for drivers that need a real draw surface, but the
  // surfaceless platform does not require one and at least the NVIDIA driver refuses to bind
  // a pbuffer here once the process already holds a GLX context (as it does whenever the
  // Simulate viewer is open): eglMakeCurrent returns false while eglGetError reports success.
  // MuJoCo only ever renders into its own offscreen framebuffer, so binding with no surface
  // at all is equivalent, and it is what the surfaceless platform is for. Try the pbuffer
  // first so drivers that do want one keep working, then fall back.
  if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_))
  {
    RCLCPP_WARN(node_->get_logger(),
                "EGL: Could not bind the PBuffer surface (error: 0x%x). Retrying surfaceless, which is "
                "sufficient because rendering targets an offscreen framebuffer.",
                eglGetError());
    if (egl_surface_ != EGL_NO_SURFACE)
    {
      eglDestroySurface(egl_display_, egl_surface_);
      egl_surface_ = EGL_NO_SURFACE;
    }
    if (eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, egl_context_))
    {
      RCLCPP_INFO(node_->get_logger(), "EGL: Successfully initialized headless OpenGL context (surfaceless)");
      return true;
    }
    RCLCPP_ERROR(node_->get_logger(),
                 "EGL: Failed to make context current (error: 0x%x). If the Simulate viewer is open, that "
                 "is the cause: this driver refuses to bind an EGL context in a process that already holds "
                 "a GLX one, and eglGetError misreports it as success. Use render_backend=egl only "
                 "together with headless, or leave render_backend at auto to render through GLFW.",
                 eglGetError());
    cleanup_egl_context();
    return false;
  }

  RCLCPP_INFO(node_->get_logger(), "EGL: Successfully initialized headless OpenGL context");
  return true;
}

void CameraPlugin::cleanup_egl_context()
{
  if (egl_display_ != EGL_NO_DISPLAY)
  {
    eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (egl_surface_ != EGL_NO_SURFACE)
    {
      eglDestroySurface(egl_display_, egl_surface_);
      egl_surface_ = EGL_NO_SURFACE;
    }
    if (egl_context_ != EGL_NO_CONTEXT)
    {
      eglDestroyContext(egl_display_, egl_context_);
      egl_context_ = EGL_NO_CONTEXT;
    }
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
  }
}

void CameraPlugin::update_loop()
{
  GLFWwindow* window = nullptr;

  if (use_egl_)
  {
    // Initialize EGL for headless rendering
    if (!init_egl_context())
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to initialize EGL context. Disabling camera publishing.");
      publish_images_ = false;
      return;
    }
  }
  else
  {
    // Use GLFW for offscreen context (display available)
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(1, 1, "", NULL, NULL);
    if (!window)
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create GLFW window. Disabling camera publishing.");
      publish_images_ = false;
      return;
    }
    glfwMakeContextCurrent(window);
  }

  // Determine rendering backend name for logging
  const char* backend = use_egl_ ? "EGL" : "GLFW";

  // Initialization of the context and data structures has to happen in the rendering thread
  RCLCPP_INFO(node_->get_logger(), "Initializing rendering for cameras (using %s)", backend);
  mjv_defaultOption(&mjv_opt_);
  mjv_defaultScene(&mjv_scn_);
  mjr_defaultContext(&mjr_con_);

  // Turn rangefinder rendering off so we don't get rays in camera images
  mjv_opt_.flags[mjtVisFlag::mjVIS_RANGEFINDER] = 0;
  // Turn off site rendering so that visualization is more realistic in cameras for testing perception.
  for (int i = 0; i < mjNGROUP; i++)
  {
    mjv_opt_.sitegroup[i] = 0;
  }

  // Initialize data for camera rendering
  mj_camera_data_ = mj_makeData(mj_model_);
  RCLCPP_INFO(node_->get_logger(), "Starting the camera rendering loop, publishing at %f Hz", camera_publish_rate_);

  // create scene and context
  mjv_makeScene(mj_model_, &mjv_scn_, 2000);

  mjr_makeContext(mj_model_, &mjr_con_, mjFONTSCALE_150);

  // Ensure the context will support the largest cameras
  int max_width = 1, max_height = 1;
  for (const auto& cam : cameras_)
  {
    max_width = std::max(max_width, static_cast<int>(cam.width));
    max_height = std::max(max_height, static_cast<int>(cam.height));
  }
  mjr_resizeOffscreen(max_width, max_height, &mjr_con_);
  RCLCPP_INFO(node_->get_logger(), "Resized offscreen buffer to %d x %d", max_width, max_height);

  // Only process images once all data has been initialized, and do it until told to stop.
  // Publishing is enabled under the lock so that a shutdown requested during the (possibly
  // slow) initialization above is observed here rather than being clobbered.
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    publish_images_ = !stop_requested_;
  }
  while (rclcpp::ok() && !stop_requested_)
  {
    std::unique_lock<std::mutex> lock(data_mutex_);

    // Wait for the main thread to copy the data and trigger this to process and publish
    // images. Note that condition_variables can be awoken spuriously, so the additional
    // checks are necessary to avoid doing work excepting when updated rendering data has
    // been made available.
    data_cv_.wait(lock, [this] { return new_data_ || stop_requested_; });

    // Shutdown triggered, kill the loop and clean up.
    if (stop_requested_)
    {
      break;
    }
    new_data_ = false;
    lock.unlock();

    update_cameras();

    // Release the snapshot so `update` may fill it again. This has to happen after
    // update_cameras() returns, because that call is what reads mj_camera_data_ and the
    // per-camera render_pending flags without holding the lock.
    {
      std::lock_guard<std::mutex> release_lock(data_mutex_);
      render_in_flight_ = false;
    }
  }
  publish_images_ = false;

  mjv_freeScene(&mjv_scn_);
  mjr_freeContext(&mjr_con_);
  mj_deleteData(mj_camera_data_);

  if (use_egl_)
  {
    cleanup_egl_context();
  }
  else if (window)
  {
    glfwDestroyWindow(window);
  }
}

void CameraPlugin::update_cameras()
{
  // Gather the cameras flagged for rendering this pass, clearing the flag as we go.
  render_indices_.clear();
  for (size_t i = 0; i < cameras_.size(); ++i)
  {
    auto& camera = cameras_[i];
    if (camera.render_pending)
    {
      camera.render_pending = false;
      render_indices_.push_back(i);
    }
  }
  if (render_indices_.empty())
  {
    return;
  }

  // Rendering is done offscreen using the data snapshot taken in `update`.
  mjr_setBuffer(mjFB_OFFSCREEN, &mjr_con_);

  camera_near_distance_ = static_cast<float>(mj_model_->vis.map.znear * mj_model_->stat.extent);
  const float far = static_cast<float>(mj_model_->vis.map.zfar * mj_model_->stat.extent);
  camera_depth_scale_ = 1.0f - camera_near_distance_ / far;

  // Use the snapshotted data's timestamp in the ROS header, as that is when the data was actually pulled.
  const rclcpp::Duration duration = rclcpp::Duration::from_seconds(mj_camera_data_->time);
  rclcpp::Time stamp(duration.nanoseconds(), RCL_ROS_TIME);

  pass_gl_seconds_ = 0.0;
  pass_convert_seconds_ = 0.0;
  pass_publish_seconds_ = 0.0;

  for (const auto idx : render_indices_)
  {
    render_and_publish_camera(cameras_[idx], stamp);
  }

  // Report the cost split. The three stages differ by more than an order of magnitude, so
  // saying which one dominates turns "cannot keep up" into an actionable message.
  //
  // Escalate to WARN only when frames were actually lost since the last report, rather than
  // when the pass outlasts 1/camera_publish_rate. The publish interval is measured on the sim
  // clock, so the wall-clock budget for a pass is 1/(rate * real-time factor): with the
  // simulation deliberately slowed a pass may take far longer than 1/rate and still drop
  // nothing. Keying off the drop counter is correct at any speed factor.
  const double pass_seconds = pass_gl_seconds_ + pass_convert_seconds_ + pass_publish_seconds_;
  const char* const fmt = "Render pass: %zu camera(s) in %.1f ms "
                          "(render+readback %.1f, convert %.1f, publish %.1f) -> ceiling %.1f Hz";
  const uint64_t drops = dropped_frames_.load();
  const bool losing_frames = drops > reported_drops_;
  reported_drops_ = drops;
  if (losing_frames)
  {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000, fmt, render_indices_.size(),
                         1e3 * pass_seconds, 1e3 * pass_gl_seconds_, 1e3 * pass_convert_seconds_,
                         1e3 * pass_publish_seconds_, 1.0 / pass_seconds);
  }
  else
  {
    RCLCPP_DEBUG_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000, fmt, render_indices_.size(),
                          1e3 * pass_seconds, 1e3 * pass_gl_seconds_, 1e3 * pass_convert_seconds_,
                          1e3 * pass_publish_seconds_, pass_seconds > 0.0 ? 1.0 / pass_seconds : 0.0);
  }
}

float CameraPlugin::modelDepth(float true_depth, const CameraData& camera)
{
  // Outside the usable range a stereo depth camera returns nothing, and the ROS
  // convention for "no data" in a 32FC1 depth image is NaN. Reporting a number here is
  // what lets a consumer silently trust geometry the real sensor could never provide:
  // MuJoCo happily returns 0.2477 m, well inside the D435i's 0.28 m Min-Z.
  if (!(true_depth >= depth_min_range_) || true_depth > depth_max_range_)
  {
    return std::numeric_limits<float>::quiet_NaN();
  }

  // Holes. Real depth drops out on low-texture, specular and occluded surfaces, none of
  // which MuJoCo knows about, so this is a blunt stand-in for their FREQUENCY rather
  // than their spatial structure -- real dropouts come in patches, these do not.
  if (depth_dropout_fraction_ > 0.0 && dropout_distribution_(noise_generator_) < depth_dropout_fraction_)
  {
    return std::numeric_limits<float>::quiet_NaN();
  }

  // Stereo triangulation error, which grows with the SQUARE of range because depth is
  // inversely proportional to disparity:
  //     sigma(Z) = Z^2 * subpixel / (focal_px * baseline)
  //
  // Derived from the geometry rather than fixed as a percentage, so it stays right if
  // the resolution or field of view changes. At the 0.15 px default, with the 50 mm
  // baseline and this rig's intrinsics, that is:
  //
  //            fx 617 (colour/depth)      fx 433 (infra)
  //   0.5 m       1.2 mm  (0.24%)          1.7 mm  (0.35%)
  //   1.0 m       4.9 mm  (0.49%)          6.9 mm  (0.69%)
  //   2.0 m      19.4 mm  (0.97%)         27.7 mm  (1.39%)
  //   3.0 m      43.7 mm  (1.46%)         62.4 mm  (2.08%)
  //
  // About 1% at 2 m, which is the typical figure reported for a D435i rather than the
  // datasheet's "<2% at 2 m" worst case. Raise depth_subpixel_error to roughly 0.3 px
  // to sit on that bound instead; the parameter is the honest place to express how
  // pessimistic you want to be.
  // The very focal length published in camera_info, so the noise matches the geometry
  // the consumer will actually reproject with.
  const float focal_px = static_cast<float>(camera.camera_info.k[0]);
  const float sigma = true_depth * true_depth * depth_subpixel_error_ /
                      std::max(1e-6f, focal_px * depth_stereo_baseline_);
  const float noisy = true_depth + sigma * static_cast<float>(noise_distribution_(noise_generator_));
  // Noise can push a sample out of range; it is still a reading the sensor would not
  // return.
  if (!(noisy >= depth_min_range_) || noisy > depth_max_range_)
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return noisy;
}

void CameraPlugin::render_and_publish_camera(CameraData& camera, const rclcpp::Time& stamp)
{
  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();

  // Step 1: Render the scene and copy images to relevant camera data containers.
  // Render scene
  mjv_updateScene(mj_model_, mj_camera_data_, &mjv_opt_, NULL, &camera.mjv_cam, mjCAT_ALL, &mjv_scn_);
  mjr_render(camera.viewport, &mjv_scn_, &mjr_con_);

  // Copy image into relevant buffers
  mjr_readPixels(camera.image_buffer.data(), camera.depth_buffer.data(), camera.viewport, &mjr_con_);

  const auto t_read = clock::now();

  // Step 2: Adjust the images and copy depth data.
  // Fix non-linear depth buffer and flip it vertically (OpenGL's origin is the bottom left)
  // https://github.com/google-deepmind/mujoco/blob/3.4.0/python/mujoco/renderer.py#L190
  auto* depth_out = reinterpret_cast<float*>(camera.depth_image.data.data());
  for (uint32_t h = 0; h < camera.height; ++h)
  {
    const float* src_row = camera.depth_buffer.data() + static_cast<size_t>(h) * camera.width;
    float* dst_row = depth_out + static_cast<size_t>(camera.height - 1 - h) * camera.width;
    for (uint32_t w = 0; w < camera.width; ++w)
    {
      const float true_depth = camera_near_distance_ / (1.0f - src_row[w] * camera_depth_scale_);
      dst_row[w] = depth_sensor_model_ ? modelDepth(true_depth, camera) : true_depth;
    }
  }

  // OpenGL's coordinate system's origin is in the bottom left, so we invert the images row-by-row
  const auto row_size = camera.width * 3;
  for (uint32_t h = 0; h < camera.height; ++h)
  {
    const auto src_idx = h * row_size;
    const auto dest_idx = (camera.height - 1 - h) * row_size;
    std::memcpy(&camera.image.data[dest_idx], &camera.image_buffer[src_idx], row_size);
  }

  const auto t_convert = clock::now();

  // Step 3: Publish the images and camera info.
  camera.image.header.stamp = stamp;
  camera.depth_image.header.stamp = stamp;
  camera.camera_info.header.stamp = stamp;

  camera.image_pub->publish(camera.image);
  camera.depth_image_pub->publish(camera.depth_image);
  camera.camera_info_pub->publish(camera.camera_info);

  const auto t_publish = clock::now();
  pass_gl_seconds_ += std::chrono::duration<double>(t_read - t_start).count();
  pass_convert_seconds_ += std::chrono::duration<double>(t_convert - t_read).count();
  pass_publish_seconds_ += std::chrono::duration<double>(t_publish - t_convert).count();
}

void CameraPlugin::handle_trigger(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> response, const int camera_idx)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    cameras_.at(camera_idx).poll_requested = true;
  }
  // Wake the fast path in `update()` so the request is picked up on the next sim step.
  poll_pending_.store(true);
  response->success = true;
}

}  // namespace mujoco_ros2_control_plugins

// Export the plugin
PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control_plugins::CameraPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
