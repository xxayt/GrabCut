

> 要求实现一篇论文“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts, ACM SIGGRAPH, 2004.
>
> https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf
>
> 实现GrabCut任务时请注意：
>
> 1. Matting部分不是GrabCut算法的精华，这部分可以不做。
> 2. 注意输出整个程序的一些重要中间结果（那些重要自己判断），面试中会有相关问题。
> 3. 如果有期末考试之类的事情，优先准备期末考试，不用急着参与测验。测验问题回答时的错误不要归因于忙期末考试所以没时间细看论文。
> 4. 面试用QQ视频共享桌面的形式。
> 5. 代码请用C++自己实现（GraphCut部分可以调用现成的函数库：https://vision.cs.uwaterloo.ca/code/ 中的Max-flow/min-cut）。图片的输入输出和用户交互部分推荐调用OpenCV。Python 语言不是不让用，但是很难写出实时运行的代码
> 6. 完成这个任务时，请思考一个问题，GMM颜色模型换成颜色直方图（可以参考这篇论文[Global contrast based salient region detection](https://mmcheng.net/salobj/) 看看怎么实现快速的彩色颜色的直方图），会对结果有什么影响。
> 7. 对于一个400*600的图像，这个程序运行时间通常是1s以内（实现的好的话，0.1s左右也很正常）。如果你的程序运行时间明显过长，请认真优化。注意测量程序执行时间请用release模式（显示，实际运行等场合用的），而不是debug模式（调试程序用的，经常比release模式慢10倍左右）。
>
> 请在收到这个考核题目10天之内联系老师面试。第一轮面试由任博，刘夏雷，或郭春乐老师进行（我已抄送各位老师）。任博老师（微信号：Rabbikun）统一安排。

## Install：Opencv(c++) & Cmake

按照[这个(VScode搭建Opencv)](https://blog.csdn.net/qq_45022687/article/details/120241068)做的，要完全按照他选的版本

## Introduction

。。。

## Result

sheep

| 迭代2次（Usum, Vsum） |                      $K=2$                       |                      $K=5$                       | $K=10$ |
| :-------------------: | :----------------------------------------------: | :----------------------------------------------: | :----: |
|     $\gamma= 10$      | （1464352, 24665）0.762s（1464226, 24738）0.368s | (1487957, 14941) 0.587s (1486784, 15332) 0.404s  |        |
|     $\gamma= 50$      | (1490879, 25298) 0.679s; (1489397, 24348) 0.227s | （1464352, 24665）0.762s（1464226, 24738）0.368s |        |
|     $\gamma= 100$     |                                                  | (1454605, 37359) 0.796s; (1453664, 36674) 0.339s |        |
|     $\gamma= 250$     |                                                  | (1448309, 83212) 0.885s;(1445567, 81238) 0.471s  |        |

bird

| 迭代2次（Usum, Vsum*50/$\gamma$） |                      K=2                       |                       K=5                       | K=10 |
| :-------------------------------: | :--------------------------------------------: | :---------------------------------------------: | :--: |
|           $\gamma= 10$            | (647359, 23497) 0.665s; (643085, 19995) 0.358s | (667382, 31275) 0.361s；(667542, 38310) 0.248sn |      |
|           $\gamma= 50$            | (640962, 19833) 0.392s; (640659, 19689) 0.286s | (663338, 23480) 0.477s；(659077, 19983) 0.302s  |      |
|           $\gamma= 100$           |                                                | (657127, 37407) 0.571s; (657131, 37586) 0.269s  |      |
|           $\gamma= 250$           |                                                | (632116, 62559) 0.725s; (603171, 37275) 0.349s  |      |

## Reference

- 论文

  1. [Interactive Graph Cutsfor Optimal Boundary & Region Segmentation  of Objects in N-D Images](https://ieeexplore.ieee.org/document/937505)
  2. ["GrabCut": interactive foreground extraction using iterated graph cuts](https://dl.acm.org/doi/10.1145/1015706.1015720)
  3. [Global contrast based salient region  detection](https://ieeexplore.ieee.org/document/6871397) 

- 笔记

  [GrabCut算法详解：从GMM模型说起_grabcut算法数学表示_lvzelong2014的博客-CSDN博客](https://blog.csdn.net/lvzelong2014/article/details/127616653)

  [读《"GrabCut" -- Interactive Foreground Extraction using Iterated Graph Cuts》 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/20255114)

  [论文阅读---“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts_Blotic，的博客-CSDN博客](https://blog.csdn.net/lmflmfa/article/details/121523272)

  [图像分割之（三）从Graph Cut到Grab Cut_zouxy09的博客-CSDN博客](https://blog.csdn.net/zouxy09/article/details/8534954)

  [图割论文阅读笔记：“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts_grabcut论文_充数的竽的博客-CSDN博客](https://blog.csdn.net/shi923281339/article/details/53383715)

  [OpenCV图像分割Grabcut算法_opencv 图像分割算法_知来者逆的博客-CSDN博客](https://blog.csdn.net/matt45m/article/details/103778232)

  [【图像处理】图像分割之（一~四）GraphCut，GrabCut函数使用和源码解读（OpenCV）_苏源流的博客-CSDN博客](https://blog.csdn.net/kyjl888/article/details/78253829)

  [代码清晰：GrabCut与BorderMatting的C++实现_border matting_蹦蹦跳跳小米粒的博客-CSDN博客](https://blog.csdn.net/weixin_41319239/article/details/91492277?spm=1001.2014.3001.5506)

  [原理清晰：GrabCut算法详解：从GMM模型说起_grabcut::buildgmms_lvzelong2014的博客-CSDN博客](https://blog.csdn.net/lvzelong2014/article/details/127616653)

  [原理清晰：From GMM to GrabCut_grabcut gmm_三分明月落的博客-CSDN博客](https://blog.csdn.net/qq_40755643/article/details/89480003?spm=1001.2014.3001.5506)

  [比较HC：四种比较简单的图像显著性区域特征提取方法原理及实现-----> AC/HC/LC/FT。 - Imageshop - 博客园 (cnblogs.com)](https://www.cnblogs.com/Imageshop/p/3889332.html)

  [HC(Histogram-based Contrast) 基于直方图对比度的显著性 - yfor - 博客园 (cnblogs.com)](https://www.cnblogs.com/yfor1008/p/15404627.html)

- 参考代码

  [MatthewLQM/GrabCut: This is an implement of GrabCut. (github.com)](https://github.com/MatthewLQM/GrabCut/tree/master)

  [MaxtirError/GrabCut: pure C++ method implement GrabCut algorithm (github.com)](https://github.com/MaxtirError/GrabCut)

  [bittnt/ImageSpirit (github.com)](https://github.com/bittnt/ImageSpirit)

- 讲解

  [GrabCut_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1w44y1f7zy/?vd_source=05f97c55a318d0682c7cce673cbb8506)

  [数字图像处理实验演示 - 43. GrabCut 人物分割与背景替换_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Bq4y1h7Zz/?spm_id_from=333.337.search-card.all.click&vd_source=05f97c55a318d0682c7cce673cbb8506)