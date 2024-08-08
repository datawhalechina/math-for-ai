
翻译：何瑞杰
# 第三章 解析几何（70）

在第二章中，我们从一般但抽象的角度研究了向量、向量空间和线性映射。在本章中，我们将从几何直觉的视角考虑这些概念。例如，我们将考虑（欧几里得空间中的）两个向量的几何表示，计算它们的长度、它们之间的距离和夹角。我们需要将向量空间装配上诱导出其几何特征的内积以完成上面所说的事情。内积及其诱导的范数和度量与我们直觉中的“相似度”和“距离”相对应；我们将在第十二章中使用它们构建支持向量机模型。随后我们将使用上面定义的向量的长度和向量间的夹角讨论正交投影。它将在第九章中的极大似然估计和第十章中的主成分分析中占中心地位。图 3.1 给出了本章中概念之间及与本书中其他章节关系的概览。

#TODO 图片汉化
![500](Pasted%20image%2020240302105358.png)

<center>图 3.1：本章中概念之间和与本书中其他章节间联系的思维导图</center>


## 3.1 范数（71）

当我们考虑几何意义下的向量，也就是原点出发的有向线段时，其长度显然是原点到有向线段终点之间的直线距离。下面我们将使用范数的概念讨论向量的长度。


> **定义 3.1**（范数）一个*范数*是向量空间$V$上的一个函数：
> $$ \begin{align} \| \cdot \|: V &\rightarrow \mathbb{R} \tag{3.1}\\ x &\mapsto \| x \|,  \tag{3.2}\end{align} $$
> 它给出每个向量空间中每个向量$x$的实值*长度*$\| x \| \in \mathbb{R}$，且对于任意的$x, y \in V$以及$\lambda \in \mathbb{R}$，满足下面的条件：
> * （绝对一次齐次）$\| \lambda x\| = |\lambda| \|x\|$，
> * （三角不等式）$\|x + y\| \leqslant \|x\| + \|y\|$，
> * （半正定）$\|x\| \geqslant 0$，当且仅当$x = 0$时取等


如图 3.2 所示，在几何中，三角不等式是说任意三角形的两边之和一定大于等于第三边。

![150](Screenshot%202024-03-02%20at%2011.10.09.png)
<center>图 3.2：三角不等式的几何表示</center>


虽然定义 3.1 考虑的是所有向量空间（2.4 节），但在本书中，我们仅考虑有限维向量空间$\mathbb{R}^{n}$。


最后别忘了，我们使用下标$i$表示$\mathbb{R}^{n}$中的向量$x$的第$i$个分量。


> **示例 3.1**（曼哈顿范数）
> $\mathbb{R}^{n}$上的*曼哈顿范数*（又叫$\mathscr{l}_{1}$范数）的定义如下：
> $$\|x\|_{1} := \sum\limits_{i=1}^{n} | x_{i} |, \tag{3.3}$$
> 其中$| \cdot |$是绝对值函数。 图 3.3 的左侧显示了平面$\mathbb{R}^{2}$上所有满足$\| x\| =  1$的点集。


> **示例 3.2** （欧几里得范数）
> 向量$x \in \mathbb{R}^{n}$的*欧几里得范数*（又叫$\mathscr{l}_{2}$范数）定义如下：
> $$ \|x\|_{2} := \sqrt{ \sum\limits_{i=1}^{n} x_{i}^{2} } = \sqrt{ x^{\top}x }, \tag{3.4}$$
> 它计算向量$x$从原点出发到终点的欧几里得距离（译者注：也就是我们通常意义下的距离）。图 3.3 的右侧显示了$\mathbb{R}^{2}$平面上所有满足$\|x\|_{2} = 1$的点集。

![600](Pasted%20image%2020240302111733.png)


<center>图 3.3：平面上满足向量在不同范数的度量下值为1的情况：左侧为曼哈顿范数，右侧为欧几里得范数</center>

注：在本书中，若不指明，范数一般是指欧几里得范数（式 3.4）。

## 3.2 内积（72）

内积的引入是后面若干几何直觉上的概念，如向量长度、向量间夹角的铺垫。

引入内积的一个主要目的是确认两个向量是否*正交*。

### 3.2.1 点积

我们已经熟悉一些特殊形式的点积，如标量积或$\mathbb{R}^{n}$中的点积，由下面的式子给出：
$$
x^{\top}y = \sum\limits_{i=1}^{n} x_{i}y_{i}. \tag{3.5}
$$


在本书中，我们称这样的内积形式为*点积*。需要注意的是，我们将介绍的内积是更加一般的概念，只要满足一些条件即可。

### 3.2.2 一般的点积

回忆在 2.7 节中提到的线性映射：我们可以利用其性质对加法和标量乘法进行重排。一个$V$上的*双线性映射*$\Omega$接受两个参数，并对其中的任意一个参数保持线性（译者注：即双重线性）。任取$x, y, z \in \Omega$，$\lambda, \psi \in \mathbb{R}$，我们有
$$
\begin{align}
\Omega(\lambda x + \psi y, z) = \lambda \cdot \Omega(x, z) + \psi \cdot \Omega(y, z), \tag{3.6}\\
\Omega(x, \lambda y + \psi z) = \lambda \cdot \Omega(x, y) + \psi \cdot \Omega(x, z). \tag{3.7}\\
\end{align}
$$
在式中，（式 3.6）表示函数对第一个变量线性；（式 3.7）表示函数对第二个变量线性（见式 2.87）。

> **定义 3.2**
> 设$V$为向量空间，双线性映射$\Omega:  V \times V \rightarrow \mathbb{R}$将两个$V$中的向量映射到一个实数，则
> * 若对所有$x, y \in V$，都有$\Omega(x, y) = \Omega(y, x)$，也即两个变量可以调换顺序，则称$\Omega$为*对称*的
> * 若对所有$x \in V$，都有
> $$\forall x \in V \textbackslash \{ 0 \}: \Omega(x, x) > 0, ~~ ~~ \Omega(0, 0) = 0,\tag{3.8}$$
>   则称$\Omega$为*正定*的。

> **定义 3.3**
> 设$V$为向量空间，双线性映射$\Omega:  V \times V \rightarrow \mathbb{R}$将两个$V$中的向量映射到一个实数，则
> * 对称且正定的双线性映射$\Omega$被称为$V$上的一个*内积*，并简写$\Omega(x, y)$为$\left\langle x, y \right\rangle$。
> * 二元组$(V, \left\langle \cdot, \cdot \right\rangle)$称为*内积空间*或*装配有内积的（实）向量空间*。特别地，如果内积采用（式 3.5）中定义的点积，则称$(V, \left\langle \cdot, \cdot \right\rangle)$为欧几里得向量空间（译者注：简称欧氏空间）

本书中我们称这些空间为内积空间。

> **示例 3.3**（不是点积的内积）
> 考虑$V = \mathbb{R}^{2}$，定义下面的内积：
> $$\left\langle x, y \right\rangle := x_{1}y_{1} - (x_{1}y_{2} + x_{2}y_{1}) + 2x_{2}y_{2},\tag{3.9}$$
> 可以验证这是一个与点积不同的内积，证明留作练习。

### 3.2.3 对称和正定矩阵

对称和正定矩阵在机器学习中十分重要，它们是由内积定义的。在 4.3 节中，我们在讨论矩阵分解时将会回到这个概念。在 12.4 节中，对称和半正定矩阵还在核的定义中起到关键作用。假设$n$维线性空间$V$装配有内积$\left\langle \cdot, \cdot \right\rangle: V \times V \rightarrow \mathbb{R}$（参见定义 3.3）并取$V$中的一个基（已排序）$B = (b_{1}, \dots, b_{n})$，在 2.6.1 节中我们知道任意$x, y \in V$，可以找到$\lambda_{i}, \psi_{i} \in \mathbb{R}, i=1,\dots,n$，使得两个向量可以写成基$B$中向量的线性组合，即$\displaystyle x = \sum\limits_{i=1}^{n} \psi_{i}b_{i} \in V$，$\displaystyle y = \sum\limits_{j=1}^{n} \lambda_{j}b_{j} \in V$。由内积的双线性性，对所有的$x, y \in V$，有

$$
\left\langle x, y \right\rangle =
\left\langle \sum\limits_{i=1}^{n} \psi_{i}b_{i}, \sum\limits_{j=1}^{n} \lambda_{j}b_{j} \right\rangle 
= \sum\limits_{i=1}^{n} \sum\limits_{j=1}^{n} \psi_{i} \lambda_{j} \left\langle b_{i}, b_{j} \right\rangle = \hat{x}^{\top} A \hat{y},
$$

其中$A_{i,j} := \left\langle b_{i}, b_{j} \right\rangle$（译者注：这就是线性空间$V$中的一个*度量矩阵*），$\hat{x}$和$\hat{y}$为原向量在基$B$下的*坐标*。这意味着内积$\left\langle \cdot, \cdot \right\rangle$被矩阵$A$*唯一确定*，且由于内积具有对称性，不难看出$A$是对称矩阵。进一步地，根据内积的正定性，我们可以得出下面的结论：
$$
\forall x \in V \textbackslash \{ 0 \}: x^{\top}Ax > 0. \tag{3.11}
$$

> **定义 3.4**（对称正定矩阵）
> 一个$n$级对称矩阵$A \in \mathbb{R}^{n \times n}$若满足（式 3.11），则被称为*对称正定矩阵*（或仅称为正定矩阵）。如果只满足将（式 3.11）中的不等号改成$\geqslant$的条件，则称为*对称半正定矩阵*


> **示例 3.4**（对称正定矩阵）
> 考虑下面两个矩阵
> $$A_{1} = \left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right] , \quad A_{2} = \left[  \begin{matrix} 9 & 6 \\ 6 & 3 \end{matrix} \right],\tag{3.12}$$
> 其中$A_{1}$是对称且正定的，因为它不仅对称（译者注：这显而易见），而且对于任意$x \in \mathbb{R}^{2} \textbackslash \{ 0 \}$都有，
> $$\begin{align} x^{\top}A_{1}x &= \left[ \begin{matrix} x_{1} & x_{2} \end{matrix}\right]\left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right]\left[ \begin{matrix} x_{1} \\ x_{2}  \end{matrix}\right] \\\ &= 9x_{1}^{2} + 12x_{1}x_{2} + 5x_{2}^{2} \\ &= (3x_{1} + 2x_{2})^{2} + x_{2}^{2} > 0.\end{align} \tag{3.13}$$
> 相反地，$A_{2}$不是正定矩阵。如果取$x = [2, -3]^{\top}$，可以验证二次型$x^\top Ax$是负数。

假设$A \in \mathbb{R}^{n \times n}$是一个对称正定矩阵，则它可以定义一个在基$B$下的内积：
$$
\left\langle x, y \right\rangle = \hat{x}^{\top}A\hat{y}, \tag{3.14}
$$
其中$x, y \in V$。

> **定理 3.5**
> 考虑一个有限维实向量空间$V$及它的一个基（有序）$B$，双线性函数$\left\langle \cdot, \cdot \right\rangle: V \times V \rightarrow R$是其上的一个内积<u>当且仅当</u>有一个对称正定矩阵$A \in \mathbb{R}^{n \times n}$，与之对应，即
> $$\left\langle x, y \right\rangle = \hat{x}^{\top} A \hat{y}. \tag{3.15}$$

下面再列举两个对称正定矩阵的性质

* 矩阵$A$的零空间（核）只包含零向量，因为若$x$不为零，则有$x^{\top}Ax>0$，于是$Ax \ne 0$。
* 矩阵$A$的所有对角元（$a_{ii}$，$i=1, \dots, n$）都是正数，因为$a_{i i} = e_{i}^{\top} Ae_{i} > 0$，其中$e$是$B$中第$i$个基向量。

## 3.3 向量长度和距离（75）
In Section 3.1, we already discussed norms that we can use to compute the length of a vector. 
在 3.1 节中，我们已经讨论过计算向量长度需要用到的范数。

Inner products and norms are closely related in the sense that any inner product induces a norm \[Formula 3.16\] in a natural way, such that we can compute lengths of vectors using the inner product. 
内积和范数这两个概念紧密相连，因为任意的内积可以自然地诱导出一个范数
$$
\|x\| := \sqrt{ \left\langle x, x \right\rangle  }, \tag{3.16}
$$

我们就可以使用内积计算向量的长度了。

However, not every norm is induced by an inner product. 
注意，不是所有范数都是由内积诱导出来的，

The Manhattan norm (3.3) is an example of a norm without a corresponding inner product. 
曼哈顿范数（式 3.3）就是一个例子。

In the following, we will focus on norms that are induced by inner products and introduce geometric concepts, such as lengths, distances, and angles.
下面我们聚焦于内积诱导的范数进行讨论，并引出相关的几何直观概念，如长度、距离和夹角。

Remark (Cauchy-Schwarz Inequality). For an inner product vector space (V,h·, ·i) the induced norm k · k satisfies the Cauchy-Schwarz inequality
注（柯西-施瓦兹不等式）：内积空间$(V, \left\langle \cdot, \cdot \right\rangle)$中由内积$\left\langle \cdot, \cdot \right\rangle$诱导的范数$\|\cdot\|$满足*柯西-施瓦兹不等式*：
$$
|\!\left\langle{x, y}\right\rangle | \leqslant \|x\| \cdot\|y\|. \tag{3.17}
$$

![500](Pasted%20image%2020240302134439.png)
> **示例 3.5**（使用内积计算向量长度）
> 在几何中，我们常关心向量的长度。现在我们可以使用内积和柯西不等式计算它们。例如取$x = [1, 1]^{\top}\in \mathbb{R}^{2}$，并令其上的内积为点积，则可以得到其长度
> $$\|x\| = \sqrt{ x^{\top}x } = \sqrt{ 1^{2} + 1^{2} } = \sqrt{ 2 }. \tag{3.18}$$
> 现在我们考虑另一个矩阵决定的内积：
> $$\left\langle x, y \right\rangle := x^{\top}\left[ \begin{matrix} 1 & -\frac{1}{2}\\ -\frac{1}{2} & 1 \end{matrix} \right]y = x_{1}y_{1} - \frac{1}{2}(x_{1}y_{2} + x_{2}y_{1}) + x_{2}y_{2} \tag{3.19}$$
> 如果我们根据这个内积的定义进行计算范数，当$x_{1}$和$x_{2}$同号时，结果会小于内积内积诱导出的范数的值，反之则会大于它。我们可以使用$x = [1, 1]^{\top}$进行实验，并发现它“看上去”比使用点积诱导出的范数的度量下要短：
> $$\left\langle x, x \right\rangle = x_{1}^{2} - x_{1}x_{2} + x_{2}^{2} = 1 - 1 + 1 = 1 \implies \|x\| = \sqrt{ 1 } = 1. \tag{3.20}$$

![500](Pasted%20image%2020240302134505.png)
![500](Pasted%20image%2020240302134537.png)
> **定义 3.6**（距离和度量）
> 考虑一个内积空间$(V, \left\langle \cdot, \cdot \right\rangle)$，任取向量$x, y \in V$，称
> $$d(x, y) := \|x - y\| = \sqrt{ \left\langle x - y, x - y \right\rangle  } \tag{3.21}$$
> 为向量$x$和$y$之间的*距离*。如果我们选用点积作为$V$上的内积，则得出的距离称为*欧几里得距离*（也称*欧氏距离*）。这样的映射
> $$\begin{align} d: V \times V & \rightarrow \mathbb{R} \tag{3.22}\\ (x, y) & \mapsto d(x, y) \tag{3.23}\end{align} x$$
> 称为*度量*。

Remark. Similar to the length of a vector, the distance between vectors does not require an inner product: a norm is sufficient. If we have a norm induced by an inner product, the distance may vary depending on the choice of the inner product. ♦
注：和向量长度类似，确定向量之间的距离不一定需要内积，使用范数足矣。如果我们有由内积有道德范数，向量间的距离因选择的内积的不同而不同。

A metric d satisfies the following:
1. d is positive definite, i.e., d(x, y) > 0 for all x, y ∈ V and d(x, y) = positive definite 0 ⇐⇒ x = y .
2. d is symmetric, i.e., d(x, y) = d(y, x) for all x, y ∈ V . symmetric triangle inequality 
3. Triangle inequality: d(x, z) 6 d(x, y) + d(y, z) for all x, y, z ∈ V .
一个度量$d$满足下面三条性质：
1. （正定性）对任意的$x, y \in V$，$d(x, y) \geqslant 0$，当且仅当$x=y$时取等，
2. （对称性）对任意的$x, y \in V$，$d(x, y) = d(y, x)$，
3. （三角不等式）对任意的$x, y, z \in V$，$d(x, y) + d(y, z) \geqslant d(x, z)$。

Remark. At first glance, the lists of properties of inner products and metrics look very similar. However, by comparing Definition 3.3 with Definition 3.6 we observe that h x, yi and d(x, y) behave in opposite directions. Very similar x and y will result in a large value for the inner product and a small value for the metric.
注：第一次看到度量的定义时，读者会发现它和内积十分相似。但如果细致比对定义 3.3 和定义 3.6，我们会发现二者的“方向”截然相反。如果两向量$x, y \in V$的内积较大，则它们之间的度量较小，反之亦然。

## 3.4 向量夹角和正交


## 3.5 正交基


## 3.6 正交补


## 3.7 函数的点积


## 3.8 正交投影
### 3.8.1 向一维子空间（直线）投影


### 3.8.2 向一般子空间投影


### 3.8.3 Gram-Schmidt 正交化


### 3.8.4 向仿射子空间投影


## 3.9 旋转
### 3.9.1 $\mathbb{R}^{2}$中的旋转


### 3.9.2 $\mathbb{R}^{3}$中的旋转


### 3.9.3 $n$维空间中的旋转


### 3.9.4 旋转算子的性质


## 3.10 拓展阅读

