include $(APPDIR)/Make.defs

PROGNAME = audiodetect
PRIORITY = $(CONFIG_AE_AUDIODETECT_PRIORITY)
STACKSIZE = $(CONFIG_AE_AUDIODETECT_STACKSIZE)
MODULE = $(CONFIG_AE_AUDIODETECT)

CSRCS = wav_reader.c audio_classifier.c alarm_report.c keyword_detect.c
ifeq ($(CONFIG_AE_AUDIODETECT_CNN),y)
CSRCS += cnn_stream_v2.c
else
CSRCS += feature_extractor.c mlp_model_infer.c
endif

ifeq ($(CONFIG_AE_AUDIODETECT_UI),y)
CSRCS += audiodetect_ui.c
endif

MAINSRC = audiodetect_main.c

include $(APPDIR)/Application.mk
