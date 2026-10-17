import pyrealsense2 as rs

# Inicializar el contexto de RealSense
ctx = rs.context()

# Obtener la lista de dispositivos conectados
devices = ctx.query_devices()

for dev in devices:
    print(f"Dispositivo: {dev.get_info(rs.camera_info.name)}")
    print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"  Firmware Version: {dev.get_info(rs.camera_info.firmware_version)}")
    print(f"  Número de Sensores: {len(dev.sensors)}")

    for sensor in dev.sensors:
        print(f"    Tipo de Sensor: {sensor.get_info(rs.camera_info.name)}")

        for option in sensor.get_supported_options():
            print(f"      Opción: {option}")

        for profile in sensor.get_stream_profiles():
            print(f"      Perfil: {profile.stream_type()} {profile.stream_index()} - {profile.format()} {profile.fps()}Hz")
