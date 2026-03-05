import ctypes

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

DXL_ID = 1
DEVICENAME = "/dev/ttyDXL_puppet_left"
BAUDRATE = 1000000

PROTOCOL_VERSION = 2.0

ADDR_TORQUE_ENABLE = 562
ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_PRESENT_POSITION = 132

TORQUE_OFF = 0
TORQUE_ON = 1


def int32(x: int) -> int:
    # convert unsigned 32-bit read to signed int32
    return ctypes.c_int32(x).value


portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

assert portHandler.openPort()
assert portHandler.setBaudRate(BAUDRATE)

# 1) torque off (EEPROM unlock)
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
    portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_OFF
)
assert dxl_comm_result == COMM_SUCCESS

# 2) read present position + current homing offset
present_u32, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
    portHandler, DXL_ID, ADDR_PRESENT_POSITION
)
homing_u32, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
    portHandler, DXL_ID, ADDR_HOMING_OFFSET
)

present = int32(present_u32)
homing = int32(homing_u32)

# 3) compute correction: want 180 deg -> 2048 ticks (single-turn)
desired_tick = 2048
delta_tick = desired_tick - present
new_homing = homing + delta_tick

print(
    f"present={present}, homing={homing}, delta={delta_tick}, new_homing={new_homing}"
)

# 4) write new homing offset
dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
    portHandler, DXL_ID, ADDR_HOMING_OFFSET, ctypes.c_uint32(new_homing).value
)
assert dxl_comm_result == COMM_SUCCESS

# 5) torque on
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
    portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ON
)
assert dxl_comm_result == COMM_SUCCESS

portHandler.closePort()
