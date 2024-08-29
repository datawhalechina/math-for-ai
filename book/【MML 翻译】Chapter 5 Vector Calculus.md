
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

所谓 Taylor 级数是将函数 $f$ 表示成的那个无限项求和式，其中的所有的项都和 $f$ 在点 $x_{0}$ 处的导数相关。

> **定义 5.3（Taylor 多项式）**
> 函数 $f: \mathbb{R} \rightarrow \mathbb{R}$ 在点 $x_{0}$ 的 $n$ 阶 Taylor 多项式是 $$T_n(x) := \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k, \tag{5.7}$$其中 $f^{(k)}(x_{0})$ 是 $f$ 在 $x_{0}$ 处的 $k$ 阶导数（假设其存在），而 $\displaystyle \frac{f^{(k)}(x_{0})}{k!}$ 是多项式各项的系数。

> 对于所有的 $t \in \mathbb{R}$ 我们约定 $t^{0} := 1$

> **定义 5.4（Taylor 级数）**
> 对于光滑函数 $f \in \mathcal{C}^{\infty}, f: \mathbb{R}\rightarrow \mathbb{R}$，它在点 $x_{0}$ 处的 Taylor 级数定义为$$T_\infty(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k. \tag{5.8}$$若 $x_{0} = 0$，我们得到了一个 Taylor 级数的特殊情况 —— Maclaurin 级数。如果 $f(x) = T_{\infty}(x)$，则我们称 $f$ 是**解析函数**。

> 注：一般而言，某个不一定为多项式函数的 $n$ 阶 Taylor 多项式是这个函数的近似，它在 $x_{0}$ 的邻域中与 $f$ 接近。事实上，对于阶数为 $k \leqslant n$ 的多项式函数 $f$，$n$ 阶 Taylor 多项式就是这个多项式函数本身，因为对所有的 $i > k$，多项式函数 $f$ 的 $i$ 阶导数 $f^{(i)}$ 均为零。

> **示例 5.3（Taylor 多项式）**
> 考虑多项式 $$f(x) = x^{4} \tag{5.9}$$并求它在 $x_{0} = 1$ 处的 Taylor 多项式 $T_{6}$。我们先求函数在该点的各阶导数 $f^{(k)}(1), k=0, \dots, 6$：$$\begin{align}f(1) &= 1 \tag{5.10} \\f'(1) &= 4 \tag{5.11} \\f''(1) &= 12 \tag{5.12} \\f^{(3)}(1) &= 24 \tag{5.13} \\f^{(4)}(1) &= 24 \tag{5.14} \\f^{(5)}(1) &= 0 \tag{5.15} \\f^{(6)}(1) &= 0 \tag{5.16}\end{align}$$于是所求的 Taylor 多项式为$$\begin{align}T_6(x) &= \sum_{k=0}^{6} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k \tag{5.17a}\\&= 1 + 4(x-1) + 6(x-1)^{2} + 4(x-1)^{3} + (x-1)^{4} + 0.\tag{5.17b}\end{align}$$整理得到$$\begin{align}T_{6}(x) &= (1-4+6-4+1) + x(4-12+12-4) \\&~ ~ ~+ x^{2}(6-12+6) + x^{3}(4-4) + x^{4}\tag{5.18a}\\&= x^{4} = f(x).\tag{5.18b}\end{align}$$
> 我们得到了和原函数一模一样的 Taylor 多项式。

![](Pasted%20image%2020240829103047.png)
<center>图 5.4</center>

> **示例 5.4（Taylor 级数）**
> 如图 5.4，考虑函数$$f(x) = \sin (x) + \cos(x) \in \mathcal{C}^{\infty}.$$我们计算其在 $x_{0} = 0$ 处的 Taylor 级数，也就是 Maclaurin 级数。我们可以求得 $f$ 在 $0$ 处的各阶导数：$$\begin{align} f(0) &= \sin(0) + \cos(0) = 1 \tag{5.20}\\ f'(0) &= \cos(0) - \sin(0) = 1 \tag{5.21}\\f''(0) &= -\sin(0) - \cos(0) = -1 \tag{5.22}\\f^{(3)}(0) &= -\cos(0) + \sin(0) = -1 \tag{5.23}\\f^{(4)}(0) &= \sin(0) + \cos(0) = f(0) = 1\tag{5.24} \\ &\vdots \end{align}$$从上面的结果我们可以找到一些规律。首先由于 $\sin(0)= 0$，有级数的各项系数只能为 $\pm {1}$，其中每个不同的值在切换为其相反数时都连续出现两次，进而有 $f^{(k+4)}(0)= f^{(k)}(0)$。
> 因此我们可以得到函数 $f$ 在 $x_{0} = 0$ 处的 Taylor 级数：$$\begin{align} T_{\infty}(x) &= \sum\limits_{k=0}^{\infty} \frac{f^{(k)}(x_{0})}{k!}(x-x_{0})^{k}\tag{5.25a}\\ &= 1 + x - \frac{1}{2!}x^{2} - \frac{1}{3!}x^{3} + \frac{1}{4!}x^{4} + \frac{1}{5!}x^{5} - \cdots\tag{5.25b}\\ &= {\color{orange} 1 - \frac{1}{2!}x^{2} + \frac{1}{4!}x^{4} \mp \cdots } + {\color{blue} x - \frac{1}{3!}x^{3} + \frac{1}{5!}x^{5} \mp \cdots }\tag{5.25c}\\ &= {\color{orange} \sum\limits_{k=0}^{\infty} \frac{(-1)^{k}}{(2k)!}x^{2k} } + {\color{blue} \sum\limits_{k=0}^{\infty} \frac{(-1)^{k}}{(2k+1)!}x^{2k+1} } \tag{5.25d} \\ &= {\color{orange} \cos(x) } + {\color{blue} \sin(x) } ,\tag{5.25e} \end{align} $$其中我们使用了三角函数的幂级数表示：$$\begin{align} \cos(x) &= \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k)!}x^{2k}, \tag{5.26}\\ \sin(x) &= \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k + 1)!}x^{2k+1}. \tag{5.27} \end{align}$$图 5.4 展示了上述条件下的前几个 Taylor 多项式 $T_{n}$，其中 $n=0,1,5,10$。

> 注：Taylor 级数是一种特殊形式的幂级数：$$f(x) = \sum\limits_{k=0}^{\infty} a_{k}(x-c)^{k},\tag{5.28}$$其中 $a_{k}$ 是系数，$c$ 是常数。不难看出这与定义 5.4 形式的一致性。

### 5.1.2 微分法则

下面我们简要介绍基本的微分法则，其中我们使用 $f'$ 表示 $f$ 的导数。
$$
\begin{align}
\text{乘法法则:}\quad & [f(x)g(x)]' = f'(x)g(x) + f(x)g'(x) \tag{5.29}\\[0.2em]
\text{除法法则:}\quad & \left[ \frac{f(x)}{g(x)} \right]' = \frac{f'(x)g(x)-f(x)g'(x)}{\big[ g(x) \big]^{2}}\tag{5.30}\\[0.2em]
\text{加法法则:}\quad & [f(x) + g(x)]' = f'(x) + g'(x) \tag{5.31}\\[0.2em]
\text{链式法则:}\quad & \Big( g\big[ f(x) \big] \Big)' = (g \circ f)'(x) = g'\big[f(x)\big]f'(x)\tag{5.32}
\end{align}
$$
其中 $g \circ f$ 表示函数的复合：$x \mapsto f(x) \mapsto g\big[f(x)\big]$。

> **示例 5.5（链式法则）**
> 使用链式法则计算函数 $h(x) = (2x+1)^{4}$ 的导数。
> 不难看出$$\begin{align}h(x) &= (2x+1)^{4} = g \big[ f(x) \big], \tag{5.33}\\ 
f(x) &= 2x+1, \tag{5.34}\\g(f) &= f^{4}, \tag{5.35}\end{align}$$然后计算 $f$ 和 $g$ 的导数：$$\begin{align}f'(x) &= 2,\tag{5.36}\\g'(f) &= 4f^{3},\tag{5.37}\\\end{align}$$这样我们就得到 $h$ 的导数：$$h'(x) {~}\mathop{=\!=\!=}\limits^{（5.32}{~} g'(f)f'(x) = (4f^{3})\cdot2 {~}\mathop{=\!=\!=}\limits^{(5.34)}{~}4(2x+1)^{3} \cdot 2 = 8(2x+1)^{3}, \tag{5.38}$$其中我们用到了链式法则，并在 $g'(f)$ 中的 $f$ 代换为 $(5.34)$ 中的表达式。


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

