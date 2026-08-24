from ament_index_python.packages import get_package_share_directory
import os
import xacro
import mujoco
import mujoco.viewer
##########################################################################################
def main():
    pkg_share_path = get_package_share_directory('my_robot_description')
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.xacro')
    world_path     = os.path.join(pkg_share_path, 'world', 'scene.xml')

    xacro_file = xacro.process_file(xacro_path)
    urdf_xml = xacro_file.toxml()
    urdf_xml = urdf_xml.replace('package://my_robot_description', pkg_share_path)

    robot_spec = mujoco.MjSpec.from_string(urdf_xml)
    world_spec = mujoco.MjSpec.from_file(world_path)
    frame = world_spec.worldbody.add_frame()
    frame.pos = [0, 0, 0.2]  # Optional: lift robot slightly above floor
    frame.attach_body(robot_spec.worldbody.bodies[0], "", "")

    model = world_spec.compile()
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
##########################################################################################
if __name__ == '__main__': main()

