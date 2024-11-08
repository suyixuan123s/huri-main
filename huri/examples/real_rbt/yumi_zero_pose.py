from huri.components.yumi_control.yumi_con import YumiController

if __name__ == "__main__":
    yumi_x = YumiController(debug=False)
    yumi_x.go_zero_pose()
    yumi_x.stop()




