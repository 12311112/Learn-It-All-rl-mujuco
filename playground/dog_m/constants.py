"""Constants for dog_m.

这个文件主要做两件事：
1. 统一保存 XML 路径、关节名、脚端名、传感器名等字符串常量；
2. 避免在 joystick.py / base.py 里到处手写名字，后续改模型时更容易同步。

注意：
这些名字必须和 XML 里的 body / joint / geom / site / sensor 的 name 完全一致。
"""

from etils import epath


# 当前 constants.py 所在目录
ROOT_PATH = epath.Path(__file__).parent

# 当前仓库中实际存在并维护的任务 XML。
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_flat_terrain.xml"

TASK_TO_XML = {
    "flat_terrain": FLAT_TERRAIN_XML,
}

AVAILABLE_TASKS = tuple(TASK_TO_XML.keys())


def task_to_xml(task_name: str) -> epath.Path:
    """根据任务名返回对应 XML 路径。

    用途：
    joystick.py 里：
        xml_path=constants.task_to_xml(task).as_posix()

    当前可选 task:
        - flat_terrain
    """
    try:
        return TASK_TO_XML[task_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported task '{task_name}'. Available tasks: {', '.join(AVAILABLE_TASKS)}"
        ) from exc


# =========================
# 四足脚端相关常量
# =========================

# 四个脚端 site 名称
# 必须对应 XML 里的：
#   <site name="FR_foot"/>
#   <site name="FL_foot"/>
#   <site name="RR_foot"/>
#   <site name="RL_foot"/>
#
# 用途：
# joystick.py 的 _post_init() 中：
#   self._feet_site_id = np.array(
#       [self._mj_model.site(name).id for name in constants.FEET_SITES]
#   )
#
# 后续在 step() 里：
#   p_f = data.site_xpos[self._feet_site_id]
# 用来取四只脚的空间位置，进一步计算摆动高度 swing_peak。
FEET_SITES = [
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
]


# 四个脚底接触 geom 名称
# 必须对应 XML 里的：
#   <geom ... name="FR_foot_bottom_tpu"/>
#   ...
#
# 用途：
# joystick.py 的 _post_init() 中：
#   self._feet_geom_id = np.array(
#       [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
#   )
#
# 后续在 reset()/step() 里通过 geoms_colliding(...)
# 判断脚底是否与地面接触，用于：
#   - contact
#   - feet_air_time
#   - last_contact
#   - first_contact
FEET_GEOMS = [
    "FR_foot_bottom_tpu",
    "FL_foot_bottom_tpu",
    "RR_foot_bottom_tpu",
    "RL_foot_bottom_tpu",
]


# 脚端位置传感器名称
# 会自动生成：
#   FR_foot_pos
#   FL_foot_pos
#   RR_foot_pos
#   RL_foot_pos
#
# 对应 XML 里的：
#   <framepos objtype="site" objname="FR_foot" name="FR_foot_pos"/>
#
# 当前你贴出来的 joystick.py 里暂时没直接用到，
# 但如果以后你想通过 sensordata 读脚端位置，而不是 site_xpos，
# 这个列表就能直接拿来用。
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]


# =========================
# 关节相关常量
# =========================

# 四个髋关节名称
# 当前 joystick.py 没直接使用这个列表，
# 但后续如果你想单独处理 hip 关节、加奖励、加限位、打印调试，会方便。
HIP_JOINT_NAMES = [
    "FR_hip_joint",
    "FL_hip_joint",
    "RR_hip_joint",
    "RL_hip_joint",
]


# 四个大腿关节名称
# 当前 joystick.py 没直接使用这个列表。
THIGH_JOINT_NAMES = [
    "FR_thigh_joint",
    "FL_thigh_joint",
    "RR_thigh_joint",
    "RL_thigh_joint",
]


# 四个小腿关节名称
# 当前 joystick.py 没直接使用这个列表。
CALF_JOINT_NAMES = [
    "FR_calf_joint",
    "FL_calf_joint",
    "RR_calf_joint",
    "RL_calf_joint",
]


# 兼容旧命名
# 有些旧代码可能还把膝关节叫 knee，这里先做个兼容映射。
# 对四足来说，你当前这个结构本质上对应的是 calf_joint。
KNEE_JOINT_NAMES = CALF_JOINT_NAMES


# 关节/执行器顺序（非常重要）
# 这个顺序必须和 XML 里的 actuator 顺序保持一致。
#
# 你的 XML actuator 顺序是：
#   FR_hip_joint
#   FR_thigh_joint
#   FR_calf_joint
#   FL_hip_joint
#   FL_thigh_joint
#   FL_calf_joint
#   RR_hip_joint
#   RR_thigh_joint
#   RR_calf_joint
#   RL_hip_joint
#   RL_thigh_joint
#   RL_calf_joint
#
# 用途：
# joystick.py 的 _post_init() 里：
#   hip_ids = [idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "hip_joint" in j]
#   thigh_ids = ...
#   calf_ids = ...
#
# 然后根据这些索引给不同关节设置不同的噪声幅度：
#   hip_pos / knee_pos / ankle_pos
#
# 这个列表如果顺序错了，噪声就会加错关节。
JOINTS_ORDER_NO_HEAD = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


# =========================
# 机身 / body 相关常量
# =========================

# 根 body 名称
# 你的 XML 里 freejoint 挂在：
#   <body name="base">
# 所以这里应该写 base。
#
# 用途：
# joystick.py 的 _post_init() 里：
#   self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
#   self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
#
# 也就是用来找到机器人根 body 的 id 和整棵子树质量。
ROOT_BODY = "trunk"


# 躯干 body 名称
# 对应 XML 里的：
#   <body name="trunk">
#
# 当前你贴出来的 joystick.py 里还没直接用到，
# 但很多时候会想单独引用躯干 body，所以先保留。
TORSO_BODY = "trunk"


# =========================
# 传感器名称常量
# =========================

# 重力方向传感器
# 对应 XML：
#   <framezaxis objtype="site" objname="imu" name="upvector"/>
#
# 用途：
# 一般在 base.py 或环境函数里读取“重力在机体坐标系下的方向”。
GRAVITY_SENSOR = "upvector"


# 世界系线速度传感器
# 对应 XML：
#   <framelinvel objtype="site" objname="imu" name="global_linvel"/>
#
# 当前 joystick.py 里通常通过 base 类函数间接读取。
GLOBAL_LINVEL_SENSOR = "global_linvel"


# 世界系角速度传感器
# 对应 XML：
#   <frameangvel objtype="site" objname="imu" name="global_angvel"/>
#
# 用途：
# joystick.py 的 _get_obs() 里：
#   global_angvel = self.get_global_angvel(data)
GLOBAL_ANGVEL_SENSOR = "global_angvel"


# 机体系线速度传感器
# 对应 XML：
#   <velocimeter site="imu" name="local_linvel"/>
#
# 用途：
# joystick.py 的奖励函数里：
#   self.get_local_linvel(data)
# 用来做 tracking_lin_vel reward
#
# 也在 _get_obs() 里读取：
#   linvel = self.get_local_linvel(data)
LOCAL_LINVEL_SENSOR = "local_linvel"


# 加速度计传感器
# 对应 XML：
#   <accelerometer site="imu" name="accelerometer"/>
#
# 用途：
# joystick.py 的 _get_obs() 里：
#   accelerometer = self.get_accelerometer(data)
# 然后加噪声后放入 state 观测。
ACCELEROMETER_SENSOR = "accelerometer"


# 陀螺仪传感器
# 对应 XML：
#   <gyro site="imu" name="gyro"/>
#
# 用途：
# joystick.py 的 _get_obs() 里：
#   gyro = self.get_gyro(data)
# 然后加噪声进观测
#
# 也在 reward 里用于 tracking_ang_vel：
#   self.get_gyro(data)
GYRO_SENSOR = "gyro"


# 前向轴传感器
# 对应 XML：
#   <framexaxis objtype="site" objname="imu" name="forwardvector"/>
#
# 当前 joystick.py 里没直接用到，先保留。
FORWARD_VECTOR_SENSOR = "forwardvector"


# IMU 在世界系下的位置传感器
# 对应 XML：
#   <framepos objtype="site" objname="imu" name="position"/>
#
# 当前 joystick.py 里没直接用到，先保留。
POSITION_SENSOR = "position"


# IMU 在世界系下的姿态四元数传感器
# 对应 XML：
#   <framequat objtype="site" objname="imu" name="orientation"/>
#
# 当前 joystick.py 里没直接用到，先保留。
ORIENTATION_SENSOR = "orientation"
