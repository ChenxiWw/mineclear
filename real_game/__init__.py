try:
    from .screen_capture import ScreenCapture
    from .mouse_controller import MouseController
    from .game_controller import RealGameController
    HAS_GUI = True
except ImportError as e:
    import warnings
    warnings.warn(f"无法加载GUI相关模块，real_game功能将不可用: {e}")
    HAS_GUI = False 