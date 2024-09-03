
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

在 5.1 节中讨论了标量元 $x \in \mathbb{R}$ 的函数 $f$ 的微分之后，本节将考虑函数 $f$ 的自变量含有多个元的一般情形，即 $\boldsymbol{x} \in \mathbb{R}^{n}$；例如 $f(x_{1}, x_{2})$。相应地，函数的导数就推广到多元情形就变成了**梯度**。

我们可以通过保持其他变量不动，然后改变变元 $x$ 来获取函数的梯度：将对各变元的偏导数组合起来。

> **定义 5.5（偏导数）**
> 给定 $n$ 元函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$，$\boldsymbol{x} \mapsto f(\boldsymbol{x}), \boldsymbol{x} \in \mathbb{R}^{n}$，它的各偏导数为$$\begin{align}\frac{ \partial f }{ \partial x_{1} } &= \lim_{ h \to 0 } \frac{f(x_{1}+h, x_{2}, \dots, x_{n}) - f(\boldsymbol{x})}{h}\\&\,\,\, \vdots\\\frac{ \partial f }{ \partial x_{n} } &= \lim_{ h \to 0 } \frac{f(x_{1}, \dots, x_{n-1}, x_{n}+h) - f(\boldsymbol{x})}{h}\end{align}\tag{5.39}$$ 然后将各偏导数组合为向量，就得到了梯度向量$$\nabla_{x}f = \text{grad} f = \frac{\mathrm{d}f}{\mathrm{d}\boldsymbol{x}} = \left[ \frac{ \partial f(\boldsymbol{x}) }{ \partial x_{1} }, \frac{ \partial f(\boldsymbol{x}) }{ \partial x_{2} }, \dots, \frac{ \partial f(\boldsymbol{x}) }{ \partial x_{n} } \right] \in \mathbb{R}^{1 \times n}, \tag{5.40}$$其中 $n$ 是变元数，$1$ 是 $f$ 像集（陪域）的维数。我们在此定义列向量 $\boldsymbol{x} = [x_{1}, \dots, x_{n}]^{\top} \in \mathbb{R}^{n}$。行向量 $(5.40)$ 称为 $f$ 的**梯度**或者**Jacobian 矩阵**，是 5.1 节中的导数的推广。

> 注：此处的 Jacobian 矩阵是其特殊情况。在 5.3 节中我们将讨论向量值函数的 Jacobian 矩阵。

> 译者注：可以看到，梯度向量是一个线性变换：$D: \mathbb{R}^{n} \rightarrow \mathbb{R}$。这样的行向量又被称为 **余向量（covector）**，其中的 余（co-）表示行和列的对偶关系。

> **示例 5.6（使用链式法则计算偏导数）**
> 给定函数 $f(x,y) = (x + 2y^{3})^{2}$，我们可以这样计算它的偏导数：$$\begin{align}\frac{ \partial f(x,y) }{ \partial x } &= 2(x+2y^{3}) \cdot \frac{ \partial  }{ \partial x } (x + 2y^{3}) = 2(x+2y^{3}), \tag{5.41} \\\frac{ \partial f(x,y) }{ \partial y } &= 2(x+2y^{3}) \cdot \frac{ \partial  }{ \partial y } (x + 2y^{3}) = 12(x+2y^{3})y^{2}. \tag{5.42}\end{align}$$上述过程中我们使用了链式法则 $(5.32)$。

> 注（作为行向量的梯度）：文献中并不常像一般的向量表示那样将梯度写为列向量。这样做的原因有两个：首先，这样的定义方便拓展为向量值函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 的情形，这样梯度就变为矩阵；其次，我们可以方便地对其使用多变元的链式法则而不用注意梯度的维数。我们将在 5.3 节中进一步讨论以上两点。

> **示例 5.7（梯度）**
> 给定函数 $f(x,y) = x_{1}^{2}x_{2} + x_{1}x_{2}^{3} \in \mathbb{R}$，它的各偏导数（相对于 $x_{1}$ 和 $x_{2}$ 求偏导）为$$\begin{align}
\frac{ \partial f(x_{1}, x_{2}) }{ \partial x_{1} } &= 2x_{1}x_{2} + x_{2}^{3} \tag{5.43}\\
\frac{ \partial f(x_{1}, x_{2}) }{ \partial x_{2} } &= x_{1}^{2} + 3 x_{1}x_{2}^{2} \tag{5.44} 
\end{align}$$于是我们可以得到梯度$$\frac{\mathrm{d}f}{\mathrm{d}\boldsymbol{x}} = \left[ \frac{ \partial f(x_{1},x_{2}) }{ \partial x_{1} } , \frac{ \partial f(x_{1}, x_{2}) }{ \partial x_{2} }  \right] = [2x_{1}x_{2} + x_{2}^{3}, x_{1}^{2} + 3x_{1}x_{2}^{2}] \in \mathbb{R}^{1 \times 2}. \tag{5.45}$$

### 5.2.1 偏导数的基本法则

当 $\boldsymbol{x} \in \mathbb{R}^{n}$ 时，即在多元函数的情况下的微分法则（如加法、乘法、链式法则）和我们在学校中学到的无异。但在对向量 $\boldsymbol{x} \in \mathbb{R}^{n}$ 求导时，我们需要额外注意，因为我们现在得到的梯度包括向量和矩阵，而矩阵乘法是非交换的。
下面是一般的加法、乘法、和链式法则：

$$
\begin{align}
\text{Product rule:}~&~\frac{ \partial  }{ \partial \boldsymbol{x} } \big[ f(\boldsymbol{x})g(\boldsymbol{x}) \big] = \frac{ \partial f }{ \partial \boldsymbol{x} }g(\boldsymbol{x}) + f(\boldsymbol{x})\frac{ \partial g }{ \partial \boldsymbol{x} } \tag{5.46}\\
\text{Sum rule:}~&~\frac{ \partial  }{ \partial \boldsymbol{ x }  } \big[ f(\boldsymbol{x}) + g(\boldsymbol{x})\big] = \frac{ \partial f }{ \partial \boldsymbol{ x }  } \frac{ \partial g }{ \partial \boldsymbol{ x }  } \tag{5.47}\\
\text{Chain rule:}~&~\frac{ \partial  }{ \partial \boldsymbol{ x }  } (g \circ f)(x) = \frac{ \partial  }{ \partial \boldsymbol{ x }  } g\big[ f(\boldsymbol{ x } ) \big] = \frac{ \partial g }{ \partial f } \frac{ \partial f }{ \partial \boldsymbol{ x }  } \tag{5.48} 
\end{align}
$$

我们仔细观察链式法则 $(5.48)$，可以通过它看到相应矩阵乘法的规律，即相邻相乘矩阵的相邻维度需要相等（见 2.2.1 节）。从左往右看，可以发现 $\partial f$ 先出现在第一项的“分母”，然后出现在第二项“分子”，按照通常乘法的定义可以理解，$\partial f$ 对应的维数对应则可以消去，剩下的就是 $\displaystyle \frac{ \partial g }{ \partial \boldsymbol{ x } }$。

> 注意，$\displaystyle \frac{ \partial f }{ \partial \boldsymbol{ x } }$ 并不是严格意义上的分数，上述说法只是为了增进理解

### 5.2.2 链式法则（chain rule）

考虑变元为 $x_{1}, x_{2}$ 函数 $f: \mathbb{R}^{2} \rightarrow \mathbb{R}$，而 $x_{1}(t)$ 和 $x_{2}(t)$ 又是变元 $t$ 的函数。为了计算 $f$ 对 $t$ 的梯度，需要用到链式法则 $(5.48)$：

$$
\frac{\mathrm{d}f}{\mathrm{d}t} = \begin{bmatrix}
\displaystyle \frac{ \partial f }{ \partial x_{1} } & \displaystyle \frac{ \partial f }{ \partial x_{2} }  
\end{bmatrix} \begin{bmatrix}
\displaystyle \frac{ \partial x_{1}(t) }{ \partial t }\\
\displaystyle \frac{ \partial x_{2}(t) }{ \partial t }\\
\end{bmatrix} = \frac{ \partial f }{ \partial x_{1} } \frac{ \partial x_{1} }{ \partial t }  + \frac{ \partial f }{ \partial x_{2} } \frac{ \partial x_{2} }{ \partial t },\tag{5.49} 
$$
其中 $\mathrm{d}$ 表示梯度，而 $\partial$ 表示偏导数。

> **示例 5.8**
> 考虑函数 $f(x_{1}, x_{2}) = x_{1}^{2} + 2x_{2}$，其中 $x_{1} = \sin t$，$x_{2} = \cos t$，则 $$\begin{align}\frac{\mathrm{d}f}{\mathrm{d}t} &= \frac{ \partial f }{ \partial x_{1} } \frac{ \partial x_{1} }{ \partial t } + \frac{ \partial f }{ \partial x_{2} } \frac{ \partial x_{2} }{ \partial t } \tag{5.50a}\\&= 2\sin t \frac{ \partial \sin t }{ \partial t } + 2 \frac{ \partial \cos t }{ \partial t } \tag{5.50b}\\&= 2\sin t \cos t - 2\sin t = 2\sin t(\cos t-1)\tag{5.50c}\end{align}$$就是 $f$ 关于 $t$ 的梯度。

如果 $f(x_{1}, x_{2})$ 是 $x_{1}$ 和 $x_{2}$ 的函数，而 $x_{1}(s, t)$ 和 $x_{2}(s,t)$ 又分别为 $s$ 和 $t$ 的函数，那么根据链式法则会得到下面的结果：

$$
\begin{align}
\frac{ \partial f }{ \partial {\color{orange} s }  } &= \frac{ \partial f }{ \partial {\color{blue} x_{1} }  } \frac{ \partial {\color{blue} x_{1} }  }{ \partial {\color{orange} s }  }  + \frac{ \partial f }{ \partial {\color{blue} x_{2} }  } \frac{ \partial {\color{blue} x_{2} }  }{ \partial {\color{orange} s }  } \tag{5.51}\\
\frac{ \partial f }{ \partial {\color{orange} t }  } &= \frac{ \partial f }{ \partial {\color{blue} x_{1} }  } \frac{ \partial {\color{blue} x_{1} }  }{ \partial {\color{orange} t }  }  + \frac{ \partial f }{ \partial {\color{blue} x_{2} }  } \frac{ \partial {\color{blue} x_{2} }  }{ \partial {\color{orange} t }  } \tag{5.52}
\end{align} 
$$
而函数的梯度为
$$
\frac{\mathrm{d}f}{\mathrm{d}(s,t)} = \frac{ \partial f }{ \partial \boldsymbol{ x }  } \frac{ \partial \boldsymbol{ x }  }{ \partial (s,t) } = \underbrace{ \begin{bmatrix}
\displaystyle \frac{ \partial f }{\color{blue} \partial x_{1} } &
\displaystyle \frac{ \partial f }{\color{orange} \partial x_{2} } 
\end{bmatrix} }_{ \displaystyle =\frac{ \partial f }{ \partial \boldsymbol{ x }  }  } \underbrace{ \begin{bmatrix}
\displaystyle {\color{blue} \frac{ \partial x_{1} }{ \partial s }  } & 
\displaystyle {\color{blue} \frac{ \partial x_{1} }{ \partial t }  } \\
\displaystyle {\color{orange} \frac{ \partial x_{2} }{ \partial s }  } & 
\displaystyle {\color{orange} \frac{ \partial x_{2} }{ \partial t }  } \\
\end{bmatrix} }_{ \displaystyle =\frac{ \partial \boldsymbol{x} }{ \partial (s,t) }  }.\tag{5.53}
$$
以上的写法 $(5.53)$ 当且仅当梯度被写为行向量时才是正确的，否则我们需要对结果进行转置，以保证矩阵的维度对应。在梯度为向量或矩阵时这样看来似乎比较显然，但当之后讨论中涉及的梯度变成 **张量（tensor）** 时对其进行转置就不那么容易了。

> **验证梯度是否正确**
> 将差商去极限而得到梯度的方法在计算机程序中的数值算法处被加以利用。当我们计算函数梯度时，我们可以通过数值的微小改变计算差商，然后校验梯度的正确性：取一个较小的值（例如 $h=10^{-4}$）然后计算有限差商和梯度的解析计算结果，如果误差足够小则说明梯度的解析结果大概率是正确的。误差足够小是指 $\displaystyle \sqrt{ \frac{\sum_{i} (dh_{i} - df_{i})^{2}}{\sum_{i} (dh_{i} + df_{i})^{2}} } < 10^{-6}$，其中 $dh_{i}$ 是指 $f$ 关于 $x_{i}$ 得到的有限差商的估计结果，$df_{i}$ 是指 $f$ 关于 $x_{i}$ 的解析梯度的计算结果。

## 5.3 向量值函数的梯度

一直以来我们讨论的都是实值函数 $f : \mathbb{R}^{n} \rightarrow \mathbb{R}$ 的偏导数和梯度，接下来我们将将此概念扩展至向量值函数（向量场）$\boldsymbol{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 的情形，其中 $n \geqslant 1, m \geqslant 1$。

给定向量值函数 $\boldsymbol{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 和向量 $\boldsymbol{x} = [x_{1}, \dots, x_{n}]^{\top}\in \mathbb{R}^{n}$，则该函数的函数值可以写为
$$
\boldsymbol{f}(\boldsymbol{x}) = \begin{bmatrix}
f_{1}(\boldsymbol{x})\\
\vdots\\
f_{m}(\boldsymbol{x})\\
\end{bmatrix} \in \mathbb{R}^{m}.\tag{5.54}
$$
这样写可以让我们将向量值函数 $\boldsymbol{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 看成一个全部由实值函数 $f_{i}: \mathbb{R}^{n} \rightarrow \mathbb{R}$  构成的向量 $[f_{1}, \dots, f_{m}]^{\top}$，而对于每一个 $f_{i}$ 我们可以不加修改的直接应用 5.2 节中的所有微分法则。这样一来，向量值函数对变元 $x_{i} \in \mathbb{R}, i=1, \dots, n$ 的偏导数由下式给出
$$
\frac{ \partial \boldsymbol{f} }{ \partial x_{i} } = \begin{bmatrix}
\displaystyle \frac{ \partial f_{1} }{ \partial x_{i} } \\
\vdots\\
\displaystyle \frac{ \partial f_{m} }{ \partial x_{i} } 
\end{bmatrix}
= 
\begin{bmatrix}
\displaystyle \lim_{ h \to 0 } \frac{f_{1}(x_{1}, \dots, x_{i-1}, x_{i}+h, x_{i+1}, x_{n}) - f_{1}(\boldsymbol{x})}{h}\\
\vdots\\
\displaystyle \lim_{ h \to 0 } \frac{f_{m}(x_{1}, \dots, x_{i-1}, x_{i}+h, x_{i+1}, x_{n}) - f_{m}(\boldsymbol{x})}{h}\\
\end{bmatrix} \in \mathbb{R}^{m}
$$


$$
\begin{align}
\frac{\mathrm{d}\boldsymbol{f}}{\mathrm{d}\boldsymbol{x}} &= 
\begin{bmatrix}
{\color{blue} \displaystyle \frac{ \partial \boldsymbol{f}(x) }{ \partial x_{1} }}  & \cdots & \color{orange} \displaystyle \frac{ \partial \boldsymbol{f}(x) }{ \partial x_{n} } 
\end{bmatrix} \tag{5.56a}\\[0.2em]
&= \begin{bmatrix}
\color{blue} \displaystyle \frac{ \partial f_{1}(\boldsymbol{x}) }{ \partial x_{1} } &\cdots &  \color{orange} \displaystyle \frac{ \partial f_{1}(\boldsymbol{x}) }{ \partial x_{n} } \\
\color{blue} \vdots & \ddots & \color{orange} \vdots\\
\color{blue} \displaystyle \frac{ \partial f_{m}(\boldsymbol{x}) }{ \partial x_{1} } & \cdots & \color{orange}  \displaystyle \frac{ \partial f_{m}(\boldsymbol{x}) }{ \partial x_{n} } \\
\end{bmatrix} \in \mathbb{R}^{m \times n} \tag{5.56b}
\end{align}
$$

$$
\begin{align}
\boldsymbol{J} &= \nabla_{x} \boldsymbol{f} = \frac{\mathrm{d}\boldsymbol{f}(\boldsymbol{x})}{\mathrm{d}\boldsymbol{x}} = \begin{bmatrix}
\displaystyle \frac{ \partial \boldsymbol{f} }{ \partial x_{1} } & \cdots & \displaystyle \frac{ \partial \boldsymbol{f} }{ \partial x_{n} } \tag{5.57}\\
\end{bmatrix}\\
&= \begin{bmatrix}
 \displaystyle \frac{ \partial f_{1}(\boldsymbol{x}) }{ \partial x_{1} } &\cdots &  \displaystyle \frac{ \partial f_{1}(\boldsymbol{x}) }{ \partial x_{n} } \\
\vdots & \ddots &  \vdots\\
\displaystyle \frac{ \partial f_{m}(\boldsymbol{x}) }{ \partial x_{1} } & \cdots & \displaystyle \frac{ \partial f_{m}(\boldsymbol{x}) }{ \partial x_{n} } \\
\end{bmatrix}, \tag{5.58}\\[0.2em]
&\quad \boldsymbol{x} = \begin{bmatrix}
x_{1}\\\vdots\\x_{n}
\end{bmatrix}, \quad J(i,j) = \frac{ \partial f_{i} }{ \partial x_{j} }. \tag{5.59} 
\end{align}
$$

$$
\begin{vmatrix}
\text{det}\left(\begin{bmatrix}
1 & 0\\0 & 1
\end{bmatrix}\right)
\end{vmatrix} = 1. \tag{5.60}
$$

$$
\begin{vmatrix}
\text{det}\left( \begin{bmatrix}
-2 & 1\\1 & 1
\end{bmatrix} \right) 
\end{vmatrix} = |-3| = 3,
\tag{5.61}
$$

$$
\boldsymbol{J} = \begin{bmatrix}
-2 & 1\\ 1 & 1
\end{bmatrix}, \tag{5.62}
$$

$$
\begin{align}
y_{1} &= -2x_{1} + x_{2} \tag{5.63}\\
y_{2} &= x_{1} + x_{2} \tag{5.64}
\end{align}
$$

$$
\frac{ \partial y_{1} }{ \partial x_{1} }  = -2, \quad \frac{ \partial y_{1} }{ \partial x_{2} } = 1, \quad \frac{ \partial y_{2} }{ \partial x_{1} } =1, \quad \frac{ \partial y_{2} }{ \partial x_{2} } = 1\tag{5.65}
$$

$$
\boldsymbol{J} = \begin{bmatrix}
\displaystyle \frac{ \partial y_{1} }{ \partial x_{1} } & \displaystyle \frac{ \partial y_{1} }{ \partial x_{2} } \\
\displaystyle \frac{ \partial y_{2} }{ \partial x_{1} } & \displaystyle \frac{ \partial y_{2} }{ \partial x_{2} }
\end{bmatrix} = 
\begin{bmatrix}
-2 & 1 \\ 1 & 1
\end{bmatrix}.
\tag{5.66}
$$


$$
\boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{Ax}, \quad \boldsymbol{f}(\boldsymbol{x}) \in \mathbb{R}^{M}, \quad \boldsymbol{A} \in \mathbb{R}^{N}.
$$

$$
f_{i}(\boldsymbol{x}) = \sum\limits_{j=1}^{N} A_{i,j}x_{j} \implies \frac{ \partial f_{i} }{ \partial x_{j} } = A_{i,j} \tag{5.67}
$$

$$
\frac{ \mathrm{d}\boldsymbol{f} }{ \mathrm{d}\boldsymbol{x} }  = \begin{bmatrix}
\displaystyle \frac{ \partial f_{1} }{ \partial x_{1} } & \cdots & 
\displaystyle \frac{ \partial f_{1} }{ \partial x_{N} } \\
\vdots & \ddots & \vdots\\
\displaystyle \frac{ \partial f_{M} }{ \partial x_{1} } & \cdots &
\displaystyle \frac{ \partial f_{M} }{ \partial x_{N } } 
\end{bmatrix} = 
\begin{bmatrix}
A_{1,1} & \cdots & A_{1,N}\\
\vdots & \ddots & \vdots\\
A_{M,1} & \cdots & A_{M,N}
\end{bmatrix} = \boldsymbol{A} \in \mathbb{R}^{M \times N}.
\tag{5.68}
$$

$$
\begin{align}
f &: \mathbb{R}^{2} \rightarrow \mathbb{R} \tag{5.69}\\
g &: \mathbb{R} \rightarrow \mathbb{R}^{2} \tag{5.70}\\
f(\boldsymbol{x}) &= \exp (x_{1}, x_{2}^{2}), \tag{5.71}\\
\boldsymbol{x} &= \begin{bmatrix}
x_{1} \\ x_{2}
\end{bmatrix} = g(t) = \begin{bmatrix}
t\cos t\\t\sin t
\end{bmatrix} \tag{5.72}
\end{align}
$$

$$
\displaystyle \frac{ \partial f }{ \partial \boldsymbol{x} } \in \mathbb{R}^{1 \times 2}, \quad \displaystyle \frac{ \partial g }{ \partial t } \in \mathbb{R}^{2 \times 1}. \tag{5.73}
$$

$$
\begin{align}
\displaystyle \frac{ \mathrm{d}h }{ \mathrm{d}t } &= {\color{blue} \displaystyle \frac{ \partial f }{ \partial \boldsymbol{x} } } {\color{orange} \displaystyle \frac{ \partial \boldsymbol{x} }{ \partial t }  } = {\color{blue} \begin{bmatrix}
\displaystyle \frac{ \partial f }{ \partial x_{1} } & \displaystyle \frac{ \partial f }{ \partial x_{2} } 
\end{bmatrix} } {\color{orange} \begin{bmatrix}
\displaystyle \frac{ \partial x_{1} }{ \partial t } \\ \displaystyle \frac{ \partial x_{2} }{ \partial t }
\end{bmatrix} }   \tag{5.74a}\\
&= 
{\color{blue} \begin{bmatrix} \exp(x_{1}x_{2}^{2})x_{2}^{2} & 2\exp(x_{1}x_{2}^{2})x_{1}x_{2}  
\end{bmatrix}}
{\color{orange} 
\begin{bmatrix}
\cos t - t\sin t\\sin t + t\cos t
\end{bmatrix}} \tag{5.74b}\\
&= \exp(x_{1}x_{2}^{2}) \big[ x_{2}^{2}(\cos t - t\sin t) + 2x_{1}x_{2}(\sin t + t\cos t) \big] , \tag{5.74c}
\end{align}
$$

$$
\boldsymbol{y} = \boldsymbol{\Phi \theta}, \tag{5.75}
$$
$$
\begin{align}
L(\boldsymbol{e}) &:= \|\boldsymbol{e}\|^{2}, \tag{5.76}\\
\boldsymbol{e}(\boldsymbol{\theta}) &:= \boldsymbol{y} - \boldsymbol{\Phi \theta}. \tag{5.77} 
\end{align}
$$
$$
\displaystyle \frac{ \partial L }{ \partial \boldsymbol{\theta} } \in \mathbb{R}^{L \times D}. \tag{5.78}
$$
$$
\displaystyle \frac{ \partial L }{ \partial \boldsymbol{\theta} } = {\color{blue} \displaystyle \frac{ \partial L }{ \partial \boldsymbol{e} }  } {\color{orange} \displaystyle \frac{ \partial \boldsymbol{e} }{ \partial \boldsymbol{\theta} }  } , \tag{5.79}
$$
$$
\displaystyle \frac{ \partial L }{ \partial \boldsymbol{\theta} } [1,d] = \sum\limits_{n=1}^{N} \displaystyle \frac{ \partial L }{ \partial \boldsymbol{e} } [n] \displaystyle \frac{ \partial \boldsymbol{e} }{ \partial \boldsymbol{\theta} } [n,d]. \tag{5.80}
$$
$$
{\color{blue} \displaystyle \frac{ \partial L }{ \partial \boldsymbol{e} }  = 2\boldsymbol{e}^{\top} } \in \mathbb{R}^{1 \times N}. \tag{5.81}
$$
$$
{\color{orange} \displaystyle \frac{ \partial \boldsymbol{e} }{ \partial \boldsymbol{\theta} } = -\boldsymbol{\Phi} } \in \mathbb{R}^{N \times D} , \tag{5.82}
$$

$$
\displaystyle \frac{ \partial L }{ \partial \boldsymbol{\theta} } = {\color{orange} - } {\color{blue} 2\boldsymbol{e}^{\top} }{\color{orange} \boldsymbol{\Phi} } {~}\mathop{=\!=\!=}\limits^{(5.77)}{~} {\color{orange} - } {\color{blue} \underbrace{ 2(\boldsymbol{y}^{\top} - \boldsymbol{\theta}^{\top}\boldsymbol{\Phi}^{\top}) }_{ 1 \times N } }~{\color{orange} \underbrace{ \boldsymbol{\Phi} }_{ N \times D } } \in \mathbb{R}^{1 \times D}. \tag{5.83}
$$
$$
L_{2}(\boldsymbol{\theta}) := \|\boldsymbol{y} - \boldsymbol{\Phi \theta}\|^{2} = (\boldsymbol{y} - \boldsymbol{\Phi \theta})^{\top}(\boldsymbol{y} - \boldsymbol{\Phi \theta}). \tag{5.84}
$$


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

## 5.6 反向传播与自动微分

### 5.6.1 深度神经网络中的梯度

### 5.6.2 自动微分

## 5.7 高阶导数

## 5.8 线性近似和多元 Taylor 级数

## 5.9 拓展阅读

## 练习

