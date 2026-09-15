# README

### Python-Trucksim多车联合仿真配置

#### 一、Trucksim配置

版本为2016.1，其余版本最多只能实现两辆车的联合仿真，是Trucksim本身的问题。python版本不限。

由于Trucksim 2016.1中本没有设置与python联仿的API接口，需要使用其与C函数交互的API接口，并且在python中调用C函数实现Trucksim与Python的联仿。

多车联合仿真主要是要配置好每辆车各自的.dll文件和.sim文件，在python中调用指定的求解器和sim文件，实现对每辆车的调用。注意.sim文件中的.dll文件路径与实际的.dll文件路径需一致。

##### 下面是具体的配置方法：

##### 1、多车模型

按照需求选择合适的车辆模型，为每辆车分别建立一个dataset，放在同一个category下面

##### 2、procedure配置

1）开环的节气门开度控制-油门，开环的制动主缸压力控制-刹车，开环的方向盘角度控制

2）运行条件选择Run forver

3）设置车辆的初始位置和速度

##### 3、Run Control配置

1）选择运行模型为：Self-Contained Solvers，选择类型为Simple C Wrapper Programme，按照默认选择外部的解释器，找到对应文件夹下的Extensions\Custom_C\solver_simple\solver_simple.exe文件作为外部解释器

2）在/extensions文件夹（或者其他文件夹下），复制多个trucksim_64.dll文件，重命名为“truck_1.dll”，“truck_2.dll”，“truck_3.dll”，放在自己建立的multi-vehicle文件夹下。这里的每个.dll文件对应的就是每辆车的求解文件

2）勾选Specific alternative VS solver file(s)，添加.dll文件路径

4）在命令行中输入Simfile xxx.sim（xxx为自定义文件名），用于关联制定车辆与仿真动画

2）配置输入分别为节气门开度，制动主缸压力，方向盘角度，配置输出为x, y, vx, vy等

##### 4、动画放映配置

在首页勾选Overlay videos and plots with other runs，添加其他车辆的dataset，这样在video&plots时能同时显示多辆车一起运行的场景

#### 二、python调用

API接口封装在Simulation.py中，通过run.py调用即可。注意在python中调用时，Trucksim必须处于打开状态，否则会提示找不到license文件

#### 三、运行联合仿真

先运行Trucksim，再运行python，运行Trucksim后出现黑窗界面，关掉即可

#### 四、参考

##### python与trucksim联合仿真参考链接：

1. https://blog.csdn.net/zataji/article/details/139455960
2. https://github.com/MizuhoAOKI/pycarsimlib

##### carsim多车仿真参考链接：

1. https://www.bilibili.com/video/BV1mp421R766/?vd_source=0e15663bdeb1dae9cce3732597438f20&spm_id_from=333.788.videopod.sections

2. https://blog.csdn.net/m0_71241814/article/details/135108311

3. https://blog.csdn.net/Libertasss/article/details/139839520?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172060259316800182793451%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=172060259316800182793451&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-139839520-null-null.142%5Ev100%5Epc_search_result_base1&utm_term=%E5%A6%82%E4%BD%95%E8%A7%A3%E5%86%B3Carsim%E5%92%8CSimulink%E5%A4%9A%E8%BD%A6%E4%BB%BF%E7%9C%9F%E6%8A%A5%E9%94%99%E7%9A%84%E9%97%AE%E9%A2%98&spm=1018.2226.3001.4187