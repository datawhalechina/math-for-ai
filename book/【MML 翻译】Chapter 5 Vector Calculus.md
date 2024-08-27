
许多机器学习算法都在优化一个目标函数，即相对于一组模型参数进行优化，这些参数控制着模型解释数据的好坏。如何寻找好的参数可被表述为一个优化问题（见 8.2 节和 8.3 节）。优化的例子包括：
1. 线性回归（见第9章），我们研究曲线拟合问题，并优化线性权重参数以最大化可能性；
2. 神经网络自编码器用于降维和数据压缩，其中参数是每层的权重和偏差，我们通过反复应用链式法则来最小化重建误差；
3. 高斯混合模型（见第11章）用于建模数据分布，我们优化每个混合组件的位置和形状参数，以最大化模型的可能性。
图5.1展示了我们通常使用利用梯度信息（第7.1节）的优化算法来解决这些问题。图5.2概述了本章概念之间以及它们与书中其他章节的联系。

本证的核心概念是函数。一个函数 $f$ 是一个数学对象，它将两个数学对象进行联系。本书中涉及的数学对象即为模型输入 $\boldsymbol{x} \in \mathbb{R}^{D}$ 以及拟合目标（函数值）$f(\boldsymbol{x})$，若无额外说明，默认拟合目标都是实数。这里 $\mathbb{R}^{D}$ 称为 $f$ 的**定义域（domain）**，而相对应的函数值 $f(\boldsymbol{x})$ 所在的集合被称为 $f$ 的**像集（image）或陪域（codomain）**。

![](Pasted%20image%2020240825122538.png)
<center>图 5.1 向量微积分在 (a) 回归问题（曲线拟合）和 (b) 分布密度估计（建模数据分布） <br>中有重要应用。</center>

![](Pasted%20image%2020240825122711.png)
<center>图 5.2 本章的概念地图及与其他章节的联系</center>

2.7.3 节中有对线性函数更为细致的讨论，但一般而言，我们将函数写为下面的形式
$$
\begin{align}
f : \mathbb{R}^D &\to \mathbb{R}\tag{5.1a}\\
\boldsymbol{x} &\mapsto f(\boldsymbol{x}) \tag{5.1b}
\end{align}
$$
其中 $(5.1a)$ 说的是 $f$ 是一个由 $\mathbb{R}^{D}$ 至 $\mathbb{R}$ 的映射，而 $(5.2b)$ 指的是 $f$ 将每一个输入 $\boldsymbol{x}$ 对应于唯一的函数值 $f(\boldsymbol{x})$。

> **示例 5.1**
> 请回忆在 3.2 节中我们谈到点积是一种特殊地内积。沿用之前的记号，函数 $f(\boldsymbol{x}) = \boldsymbol{x}^{\top}\boldsymbol{x}, \boldsymbol{x} \in \mathbb{R}^{2}$ 相当于
> $$\begin{align}f: \mathbb{R}^{2} & \rightarrow \mathbb{R}\tag{5.2a}\\\boldsymbol{x} &\mapsto x_{1}^{2} + x_{2}^{2}. \tag{5.2b}\end{align}$$

本章将介绍如何计算函数的梯度——这在机器学习中如何充分利用*学习*非常重要，因为梯度指向函数值提升的最陡峭方向。所以向量微积分是机器学习中所需的基础数学工具。我们在全书中都默认函数是可微的，但若具备一些尚未提及的额外定义，很多机器学习方法可被扩展至**次梯度（sub-differentials，当函数连续但在某些点不可微时）**。我们将在第七章探讨带有条件限制的此类函数。

## 5.1 一元函数的微分

接下来我们简要复习一下学过高中数学读者较为熟悉的一元函数的微分。我们从用于定义微分的重要概念——一元函数 $y = f(x), x,y \in \mathbb{R}$ 的差商开始。

![](Pasted%20image%2020240827204933.png)
<center>图 5.3 </center>

> **定义 5.1 (差商）**
> 一元函数的差商$$\frac{\delta y}{\delta x} := \frac{f(x + \delta x) - f(x)}{\delta x} \tag{5.3}$$
> 计算连接函数 $f$ 之图像上的两点的割线之斜率。如图 5.3 所示，两点的横坐标分别为 $x_{0}$ 和 $x_{0} + \delta x$。

若 $f$ 是线性函数，差商也可以看做函数 $f$ 上从点 $x$ 至 $x + \delta x$ 之间的平均斜率。若对 $\delta x$ 去极限 $\delta x \rightarrow 0$，我们得到了 $f$ 在 $x$ 处的切线斜率；如果 $f$ 可微，这个切线斜率就是 $f$ 在 $x$ 处的导数。

> **定义 5.2 （导数）**
> 对正实数 $h > 0$，函数 $f$ 在 $x$ 处的导数由下面的极限定义：$$\frac{\mathrm{d}f}{\mathrm{d}x} := \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} \tag{5.4}$$
> 对应到图 5.3 中，割线将变为切线。

$f$ 的导数时刻指向 $f$ 提升最快的方向。

> **示例 5.2 （多项式函数的导数）**
> 我们想计算多项式函数 $f(x) = x^{n}, n \in \mathbb{N}$ 的导数。即使我们可能已经知道答案是 $n x^{n-1}$，但我们依然希望使用导数和差商的定义导出它。
> 使用导数的定义 $(5.4)$，我们有$$\begin{align}\frac{\mathrm{d}f}{\mathrm{d}x} &= \lim_{h \to 0} \frac{{\color{blue} f(x+h)} - {\color{orange} f(x)}}{h} \tag{5.5a}\\&= \lim_{h \to 0} \frac{{\color{blue} (x + h)^n} - {\color{orange} x^n}}{h} \tag{5.5b}  \\&= \lim_{h \to 0} \frac{{\color{blue}\sum\limits_{i=0}^{n} \binom{n}{i} x^{n-i} h^i} - {\color{orange}x^n}}{h}. \tag{5.5c}\end{align}$$
> 注意到 $\displaystyle {\color{orange} x^{n}} = \binom{n}{i} x^{n-0}h^{0}$，所以上式分子相当于求和项从 $1$ 开始，于是上式变为$$\begin{align} \frac{\mathrm{d}f}{\mathrm{d}x} &= \lim_{ h \to 0 } \frac{\sum\limits_{i=1}^{n} \binom{n}{i} x^{n-i}h^{i}}{h} \tag{5.6a}\\ &= \lim_{ h \to 0 } \sum\limits_{i=1}^{n} \binom{n}{i} x^{n-i}h^{i-1} \tag{5.6b}\\ &= \lim_{ h \to 0 } \left( \binom{n}{1} x^{n-1} + \underbrace{\sum\limits_{i=2}^{n} \binom{n}{i} x^{n-i}h^{i-1}}_{\to 0 \text{ as } h \rightarrow 0} \right) \tag{5.6c}\\ &= \frac{n!}{1!(n-1)!}x^{n-1} = nx^{n-1}. \tag{5.6d} \end{align}$$

### 5.1.1 Taylor 级数

$$
T_n(x) := \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k \tag{5.7}
$$

$$
T_\infty(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k \tag{5.8}
$$

$$
\begin{align}
f^{(1)} &= 1 \tag{5.10} \\
f'(1) &= 4 \tag{5.11} \\
f''(1) &= 12 \tag{5.12} \\
f^{(3)}(1) &= 24 \tag{5.13} \\
f^{(4)}(1) &= 24 \tag{5.14} \\
f^{(5)}(1) &= 0 \tag{5.15} \\
f^{(6)}(1) &= 0 \tag{5.16}
\end{align}
$$

$$
T_6(x) = \sum_{k=0}^{6} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k \tag{5.17a}
$$

$$
T_6(x) = (1 - 4 + 6 - 4 + 1) + x(4 - 12 + 12 - 4) + x^2(6 - 12 + 6) + x^3(4 - 4) + x^4 \tag{5.17b}
$$

$$
T_6(x) = x^4 = f(x) \tag{5.18b}
$$


$$
T_6(x) = x^4 = f(x), \quad \text{where we used the power series representations} \tag{5.18b}
$$

$$
\cos(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k)!}x^{2k}, \quad \sin(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k + 1)!}x^{2k+1} \tag{5.25e}
$$

### 5.1.2 微分法则

## 5.2 偏导数和梯度

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x)}{h} \tag{5.39}
$$

$$
\begin{align}
\frac{\partial f}{\partial x_1} &= 2(x + 2y^3), \quad \ldots \tag{5.41} \\
\frac{\partial f}{\partial y} &= \frac{1}{2}(x + 2y^3)y^2 \tag{5.42}
\end{align}
$$

$$
\frac{\partial f}{\partial x_1} = 2x_1x_2 + x_3^2, \quad \frac{\partial f}{\partial x_2} = x_1^2 + 3x_1x_2^2 \tag{5.43-5.44}
$$

### 5.2.1 偏导数的基本法则

$$
\begin{align}
\left( \frac{\partial f}{\partial x} g(x) + f(x) \frac{\partial g}{\partial x} \right)' &= \frac{\partial}{\partial x} (fg) = \frac{\partial f}{\partial x} g(x) + f(x) \frac{\partial g}{\partial x} \tag{5.46} \\
\left( \frac{\partial f}{\partial x} + g(x) \right)' &= \frac{\partial}{\partial x} (f(x) + g(x)) = \frac{\partial f}{\partial x} + \frac{\partial g}{\partial x} \tag{5.47}
\end{align}
$$

$$
\frac{\partial}{\partial x} (g \circ f)(x) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial x} \tag{5.48}
$$

### 5.2.2 链式法则（chain rule）

$$
\frac{df}{dt} = \frac{\partial f}{\partial x_1} \frac{dx_1}{dt} + \frac{\partial f}{\partial x_2} \frac{dx_2}{dt} \tag{5.49}
$$

$$
\begin{align}
\frac{df}{dt} &= 2 \sin t \frac{d}{dt} (\sin t) + 2 \frac{d}{dt} (\cos t) \tag{5.50a} \\
&= 2 \sin t \cos t - 2 \sin t = 2 \sin t(\cos t - 1) \tag{5.50c}
\end{align}
$$

$$
\begin{align}
\frac{\partial f}{\partial s} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial s} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial s} \tag{5.51} \\
\frac{\partial f}{\partial t} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t} \tag{5.52}
\end{align}
$$

$$
\frac{df}{ds, t} = \frac{\partial f}{\partial x} \begin{bmatrix} \frac{\partial x_1}{\partial s} & \frac{\partial x_1}{\partial t} \\ \frac{\partial x_2}{\partial s} & \frac{\partial x_2}{\partial t} \end{bmatrix} \tag{5.53}
$$

## 5.3 向量函数的梯度


## 5.4 矩阵的梯度


## 5.5 常用梯度恒等式

$$
\begin{align}
\frac{ \partial }{ \partial \boldsymbol{X} } \boldsymbol{f}(\boldsymbol{X})^{\top} &= \left( \frac{ \partial \boldsymbol{f}(\boldsymbol{X}) }{ \partial \boldsymbol{X} }  \right)^{\top}, \tag{5.99}\\[0.2em]
\frac{ \partial  }{ \partial \boldsymbol{X} } \text{tr}\big[ \boldsymbol{f}(\boldsymbol{X}) \big] &= \text{tr}\left[ \frac{ \partial \boldsymbol{f}(\boldsymbol{X}) }{ \partial \boldsymbol{X} }  \right], \tag{5.100}\\[0.2em]
\frac{ \partial  }{ \partial \boldsymbol{X} } \det \big[ \boldsymbol{f}(\boldsymbol{X}) \big] &= \det \big[ \boldsymbol{f}(\boldsymbol{X}) \big] \text{tr}\left[ \boldsymbol{f}(\boldsymbol{X})^{-1} \frac{ \partial \boldsymbol{f}(\boldsymbol{X}) }{ \partial \boldsymbol{X} }  \right], \tag{5.101}\\[0.2em]
\frac{ \partial  }{ \partial \boldsymbol{X} } \boldsymbol{f}(\boldsymbol{X})^{-1} &= -\boldsymbol{f}(\boldsymbol{X})^{-1} \left[ \frac{ \partial \boldsymbol{f}(\boldsymbol{X}) }{ \partial \boldsymbol{X} }  \right]\boldsymbol{f}(\boldsymbol{X})^{-1} \tag{5.102}\\[0.2em]
\frac{ \partial \boldsymbol{a}^{\top}\boldsymbol{X}^{-1}\boldsymbol{b} }{ \partial \boldsymbol{X} } &= -(\boldsymbol{X}^{-1})^{\top}\boldsymbol{a}\boldsymbol{b}^{\top}(\boldsymbol{X}^{-1})^{\top}\tag{5.103}\\[0.2em]
\frac{ \partial \boldsymbol{x}^{\top}\boldsymbol{a} }{ \partial \boldsymbol{x} } &= \boldsymbol{a}^{\top}\tag{5.104}\\[0.2em]
\frac{ \partial \boldsymbol{a}^{\top}\boldsymbol{x} }{ \partial \boldsymbol{x} } &= \boldsymbol{a}^{\top}\tag{5.105}\\[0.2em]
\frac{ \partial \boldsymbol{a}^{\top}\boldsymbol{X}\boldsymbol{b} }{ \partial \boldsymbol{X} } &= \boldsymbol{a}\boldsymbol{b}^{\top} \tag{5.106}\\[0.2em]
\frac{ \partial \boldsymbol{x}^{\top}\boldsymbol{B}\boldsymbol{x} }{ \partial \boldsymbol{x} } &= \boldsymbol{x}^{\top}(\boldsymbol{B} + \boldsymbol{B}^{\top})\tag{5.107}\\[0.2em]
\frac{ \partial  }{ \partial \boldsymbol{s} } (\boldsymbol{x} - \boldsymbol{A}\boldsymbol{s})^{\top}\boldsymbol{W}(\boldsymbol{x} - \boldsymbol{A}\boldsymbol{s}) &= -2(\boldsymbol{x} - \boldsymbol{A}\boldsymbol{s})^{\top}\boldsymbol{W}\boldsymbol{A}, \tag{5.108}\\[0.2em]
& \quad \quad \text{for symmetric }\boldsymbol{W}
\end{align}
$$

