import os
import sys


def has_graphical_linux_session(environ):
    return bool(environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"))


def should_force_dummy_video_driver(platform_name=None, environ=None):
    platform_name = (platform_name or sys.platform).lower()
    environ = os.environ if environ is None else environ

    if environ.get("SDL_VIDEODRIVER"):
        return False

    if platform_name.startswith("linux"):
        return not has_graphical_linux_session(environ)

    return False


def configure_sdl_video_driver(platform_name=None, environ=None):
    environ = os.environ if environ is None else environ

    if should_force_dummy_video_driver(platform_name=platform_name, environ=environ):
        environ["SDL_VIDEODRIVER"] = "dummy"

    return environ.get("SDL_VIDEODRIVER")
