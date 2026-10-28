# CMakeLists.txt for audiodetect app (OpenVela / NuttX CMake build)
# Shared Tiny MLP classifier for CLI and UI.

if(CONFIG_AE_AUDIODETECT)
  set(AUDIODETECT_SRCS
      audiodetect_main.c
      wav_reader.c
      audio_classifier.c
      alarm_report.c
      keyword_detect.c)

  list(APPEND AUDIODETECT_SRCS feature_extractor.c mlp_model_infer.c)

  if(CONFIG_AE_AUDIODETECT_UI)
    list(APPEND AUDIODETECT_SRCS audiodetect_ui.c)
    set(AUDIODETECT_DEPENDS lvgl)
  endif()

  nuttx_add_application(
    NAME
    ${CONFIG_AE_AUDIODETECT_PROGNAME}
    MODULE
    ${CONFIG_AE_AUDIODETECT}
    SRCS
    ${AUDIODETECT_SRCS}
    DEPENDS
    ${AUDIODETECT_DEPENDS}
    STACKSIZE
    ${CONFIG_AE_AUDIODETECT_STACKSIZE}
    PRIORITY
    ${CONFIG_AE_AUDIODETECT_PRIORITY})
endif()
