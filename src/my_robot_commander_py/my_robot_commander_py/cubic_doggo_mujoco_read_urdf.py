from ament_index_python.packages import get_package_share_directory
import os, re, tempfile
import mujoco, mujoco.viewer
#############################################################################################################################
def main():
    pkg_share_path = get_package_share_directory('my_robot_description')
    urdf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.urdf')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.xml')

    with open(urdf_path, 'r') as fileObj:
        urdf_raw = fileObj.read()
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
    mjcf_content = mjcf_content.replace('<include file="cubic_doggo.mujoco.urdf"/>',robot_bodies)

    model = mujoco.MjModel.from_xml_string(mjcf_content)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.viewer.launch(model, data)
#############################################################################################################################
if __name__ == '__main__': main()

