from ament_index_python.packages import get_package_share_directory
import xacro
import os, re, time, tempfile
import mujoco, mujoco.viewer
#############################################################################################################################
def main():
    pkg_share_path = get_package_share_directory('my_robot_description')
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_leg1.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_leg1.mujoco.xml')
    usdf_file      =                                      'cubic_leg1.mujoco.urdf'

    xacro_raw = xacro.process_file(xacro_path)
    urdf_raw  = xacro_raw.toxml()
    urdf_content = urdf_raw.replace('package://my_robot_description', pkg_share_path)
    urdf_content = urdf_content.replace('</robot>', '<mujoco><compiler discardvisual="false"/></mujoco></robot>')
    urdf_content_temp = mujoco.MjModel.from_xml_string(urdf_content)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml') as tempfileObj:
        mujoco.mj_saveLastXML(tempfileObj.name, urdf_content_temp)
        tempfileObj.seek(0)
        urdf_content = tempfileObj.read()
    robot_assets = re.search(r'<asset>(.*?)</asset>',         urdf_content, re.DOTALL).group(1)
    robot_bodies = re.search(r'<worldbody>(.*?)</worldbody>', urdf_content, re.DOTALL).group(1)

    with open(mjcf_path, 'r') as fileObj:
        mjcf_content = fileObj.read()
    mjcf_content = mjcf_content.replace('</asset>', robot_assets + '\n    </asset>')
    mjcf_content = mjcf_content.replace('<include file=\"'+usdf_file+'\"/>', robot_bodies)
    mjcf_content = mjcf_content.replace('name="calfSphere_FL"', 'name="calfSphere_FL" class="foot_friction"')
    print(mjcf_content)

    model = mujoco.MjModel.from_xml_string(mjcf_content)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
#############################################################################################################################
if __name__ == '__main__': main()

