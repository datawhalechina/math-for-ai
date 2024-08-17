翻译：何瑞杰

# 第三章 解析几何（70）

In Chapter 2, we studied vectors, vector spaces, and linear mappings at a general but abstract level.
在第二章中，我们从一般但抽象的角度研究了向量、向量空间和线性映射。在本章中，我们将从几何直觉的视角考虑这些概念。

In particular, we will look at geometric vectors and compute their lengths and distances or angles between two vectors. 
例如，我们将考虑（欧几里得空间中的）两个向量的几何表示，计算它们的长度、它们之间的距离和夹角。

To be able to do this, we equip the vector space with an inner product that induces the geometry of the vector space. 
我们需要将向量空间装配上诱导出其几何特征的内积以完成上面所说的事情。

Inner products and their corresponding norms and metrics capture the intuitive notions of similarity and distances, which we use to develop the support vector machine in Chapter 12. 
内积及其诱导的范数和度量与我们直觉中的“相似度”和“距离”相对应；我们将在第十二章中使用它们构建支持向量机模型。

We will then use the concepts of lengths and angles between vectors to discuss orthogonal projections, which will play a central role when we discuss principal component analysis in Chapter 10 and regression via maximum likelihood estimation in Chapter 9. 
随后我们将使用上面定义的向量的长度和向量间的夹角讨论正交投影。它将在第九章中的极大似然估计和第十章中的主成分分析中占中心地位。

Figure 3.1 gives an overview of how concepts in this chapter are related and how they are connected to other chapters of the book.
图 3.1 给出了本章中概念之间及与本书中其他章节关系的概览。

#TODO 图片汉化

![500](Pasted%20image%2020240302105358.png)

<center>Figure 3.1 A mind map of the concepts introduced in this chapter, <br>
along with when they are used in other parts of the book.</center>
<center>图 3.1：本章中概念之间和与本书中其他章节间联系的思维导图</center>

## 3.1 范数（71）

When we think of geometric vectors, i.e., directed line segments that start at the origin, then intuitively the length of a vector is the distance of the “end” of this directed line segment from the origin. 
当我们考虑几何意义下的向量，也就是原点出发的有向线段时，其长度显然是原点到有向线段终点之间的直线距离。

In the following, we will discuss the notion of the length of vectors using the concept of a norm.
下面我们将使用范数的概念讨论向量的长度。

![500](Pasted%20image%2020240302110312.png)

> **定义 3.1**（范数）一个*范数*是向量空间$V$上的一个函数：
> $$ \begin{align} \| \cdot \|: V &\rightarrow \mathbb{R} \tag{3.1}\\ x &\mapsto \| x \|, \tag{3.2}\end{align} $$
> 它给出每个向量空间中每个向量$x$的实值*长度*$\| x \| \in \mathbb{R}$，且对于任意的$x, y \in V$以及$\lambda \in \mathbb{R}$，满足下面的条件：
> * （绝对一次齐次）$\| \lambda x\| = |\lambda| \|x\|$，
> * （三角不等式）$\|x + y\| \leqslant \|x\| + \|y\|$，
> * （半正定）$\|x\| \geqslant 0$，当且仅当$x = 0$时取等

In geometric terms, the triangle inequality states that for any triangle, the sum of the lengths of any two sides must be greater than or equal to the length of the remaining side; see Figure 3.2 for an illustration.
如图 3.2 所示，在几何中，三角不等式是说任意三角形的两边之和一定大于等于第三边。

![150](Screenshot%202024-03-02%20at%2011.10.09.png)

<center>Figure 3.2 Triangle inequality.</center>
<center>图 3.2：三角不等式的几何表示</center>

Definition 3.1 is in terms of a general vector space V (Section 2.4), but in this book we will only consider a finite-dimensional vector space Rn.
虽然定义 3.1 考虑的是所有向量空间（2.4 节），但在本书中，我们仅考虑有限维向量空间$\mathbb{R}^{n}$。

Recall that for a vector x ∈ Rn we denote the elements of the vector using a subscript, that is, xi is the ith element of the vector x.
最后别忘了，我们使用下标$i$表示$\mathbb{R}^{n}$中的向量$x$的第$i$个分量。

![500](Pasted%20image%2020240302111756.png)

> **示例 3.1**（曼哈顿范数）
> $\mathbb{R}^{n}$上的*曼哈顿范数*（又叫$\mathscr{l}_{1}$范数）的定义如下：
> $$\|x\|_{1} := \sum\limits_{i=1}^{n} | x_{i} |, \tag{3.3}$$
> 其中$| \cdot |$是绝对值函数。 图 3.3 的左侧显示了平面$\mathbb{R}^{2}$上所有满足$\| x\| =  1$的点集。

![500](Pasted%20image%2020240302112542.png)

> **示例 3.2** （欧几里得范数）
> 向量$x \in \mathbb{R}^{n}$的*欧几里得范数*（又叫$\mathscr{l}_{2}$范数）定义如下：
> $$ \|x\|_{2} := \sqrt{ \sum\limits_{i=1}^{n} x_{i}^{2} } = \sqrt{ x^{\top}x }, \tag{3.4}$$
> 它计算向量$x$从原点出发到终点的欧几里得距离（译者注：也就是我们通常意义下的距离）。图 3.3 的右侧显示了$\mathbb{R}^{2}$平面上所有满足$\|x\|_{2} = 1$的点集。

![600](Pasted%20image%2020240302111733.png)

<center>Figure 3.3 For different norms, the red lines indicate the set of vectors with norm 1.<br> Left: Manhattan norm; Right: Euclidean distance.</center>
<center>图 3.3：平面上满足向量在不同范数的度量下值为1的情况：左侧为曼哈顿范数，右侧为欧几里得范数</center>

注：在本书中，若不指明，范数一般是指欧几里得范数（式 3.4）。

## 3.2 内积（72）

Inner products allow for the introduction of intuitive geometrical concepts, such as the length of a vector and the angle or distance between two vectors. 
内积的引入是后面若干几何直觉上的概念，如向量长度、向量间夹角的铺垫。

A major purpose of inner products is to determine whether vectors are orthogonal to each other.
引入内积的一个主要目的是确认两个向量是否*正交*。

### 3.2.1 点积

We may already be familiar with a particular type of inner product, the scalar product/dot product in Rn, which is given by
我们已经熟悉一些特殊形式的点积，如标量积或$\mathbb{R}^{n}$中的点积，由下面的式子给出：
$$
x^{\top}y = \sum\limits_{i=1}^{n} x_{i}y_{i}. \tag{3.5}
$$

We will refer to this particular inner product as the dot product in this book. However, inner products are more general concepts with specific properties, which we will now introduce
在本书中，我们称这样的内积形式为*点积*。需要注意的是，我们将介绍的内积是更加一般的概念，只要满足一些条件即可。

### 3.2.2 一般的点积

> #HELP 感觉怪怪的
> Recall the linear mapping from Section 2.7, where we can rearrange the mapping with respect to addition and multiplication with a scalar. 
> 回忆在 2.7 节中提到的线性映射：我们可以利用其性质对加法和标量乘法进行重排。

A bilinear mapping linear mapping Ω is a mapping with two arguments, and it is linear in each argument, i.e., when we look at a vector space V then it holds that for all x, y, z ∈ V, λ, ψ ∈ R that
一个$V$上的*双线性映射*$\Omega$接受两个参数，并对其中的任意一个参数保持线性（译者注：即双重线性）。任取$x, y, z \in \Omega$，$\lambda, \psi \in \mathbb{R}$，我们有
$$
\begin{align}
\Omega(\lambda x + \psi y, z) = \lambda \cdot \Omega(x, z) + \psi \cdot \Omega(y, z), \tag{3.6}\\
\Omega(x, \lambda y + \psi z) = \lambda \cdot \Omega(x, y) + \psi \cdot \Omega(x, z). \tag{3.7}\\
\end{align}
$$
Here, (3.6) asserts that Ω is linear in the first argument, and (3.7) asserts that Ω is linear in the second argument (see also (2.87)).
在式中，（式 3.6）表示函数对第一个变量线性；（式 3.7）表示函数对第二个变量线性（见式 2.87）。

![500](Pasted%20image%2020240302120456.png)

> **定义 3.2**
> 设$V$为向量空间，双线性映射$\Omega:  V \times V \rightarrow \mathbb{R}$将两个$V$中的向量映射到一个实数，则
> * 若对所有$x, y \in V$，都有$\Omega(x, y) = \Omega(y, x)$，也即两个变量可以调换顺序，则称$\Omega$为*对称*的
> * 若对所有$x \in V$，都有
> $$\forall x \in V \textbackslash \{ 0 \}: \Omega(x, x) > 0, ~~ ~~ \Omega(0, 0) = 0, \tag{3.8}$$
> 则称$\Omega$为*正定*的。

![500](Pasted%20image%2020240302120513.png)

> **定义 3.3**
> 设$V$为向量空间，双线性映射$\Omega:  V \times V \rightarrow \mathbb{R}$将两个$V$中的向量映射到一个实数，则
> * 对称且正定的双线性映射$\Omega$叫做$V$上的一个*内积*，并简写$\Omega(x, y)$为$\left\langle x, y \right\rangle$。
> * 二元组$(V, \left\langle \cdot, \cdot \right\rangle)$称为*内积空间*或*装配有内积的（实）向量空间*。特别地，如果内积采用（式 3.5）中定义的点积，则称$(V, \left\langle \cdot, \cdot \right\rangle)$为欧几里得向量空间（译者注：简称欧氏空间）

We will refer to these spaces as inner product spaces in this book.
本书中我们称这些空间为内积空间。

![500](Pasted%20image%2020240302120624.png)

> **示例 3.3**（不是点积的内积）
> 考虑$V = \mathbb{R}^{2}$，定义下面的内积：
> $$\left\langle x, y \right\rangle := x_{1}y_{1} - (x_{1}y_{2} + x_{2}y_{1}) + 2x_{2}y_{2}, \tag{3.9}$$
> 可以验证这是一个与点积不同的内积，证明留作练习。

### 3.2.3 对称和正定矩阵

Symmetric, positive definite matrices play an important role in machine learning, and they are defined via the inner product. 
对称和正定矩阵在机器学习中十分重要，它们是由内积定义的。

In Section 4.3, we will return to symmetric, positive definite matrices in the context of matrix decompositions. 
在 4.3 节中，我们在讨论矩阵分解时将会回到这个概念。

The idea of symmetric positive semidefinite matrices is key in the definition of kernels (Section 12.4).
在 12.4 节中，对称和半正定矩阵还在核的定义中起到关键作用。

Consider an n-dimensional vector space V with an inner product h·, ·i :× V → R (see Definition 3.3) and an ordered basis B = (b1, ... , bn) of V . 
假设$n$维线性空间$V$装配有内积$\left\langle \cdot, \cdot \right\rangle: V \times V \rightarrow \mathbb{R}$（参见定义 3.3）并取$V$中的一个基（已排序）$B = (b_{1}, \dots, b_{n})$，

Recall from Section 2.6.1 that any vectors x, y ∈ V can be written as linear combinations of the basis vectors so that x = P n i=1 ψibi ∈ V and y = P n j=1 λjbj ∈ V for suitable ψi , λj ∈ R. 
在 2.6.1 节中我们知道任意$x, y \in V$，可以找到$\lambda_{i}, \psi_{i} \in \mathbb{R}, i=1, \dots, n$，使得两个向量可以写成基$B$中向量的线性组合，即$\displaystyle x = \sum\limits_{i=1}^{n} \psi_{i}b_{i} \in V$，$\displaystyle y = \sum\limits_{j=1}^{n} \lambda_{j}b_{j} \in V$。

Due to the bilinearity of the inner product, it holds for all $x, y ∈ V$ that
由内积的双线性性，对所有的$x, y \in V$，有
$$
\left\langle x, y \right\rangle =
\left\langle \sum\limits_{i=1}^{n} \psi_{i}b_{i}, \sum\limits_{j=1}^{n} \lambda_{j}b_{j} \right\rangle 
= \sum\limits_{i=1}^{n} \sum\limits_{j=1}^{n} \psi_{i} \lambda_{j} \left\langle b_{i}, b_{j} \right\rangle = \hat{x}^{\top} A \hat{y}, 
$$

where Aij := h bi, bj i and ˆx, ˆy are the coordinates of x and y with respect to the basis B
其中$A_{i, j} := \left\langle b_{i}, b_{j} \right\rangle$（译者注：这就是线性空间$V$中的一个*度量矩阵*），$\hat{x}$和$\hat{y}$为原向量在基$B$下的*坐标*。

 This implies that the inner product h·, ·i is uniquely determined through A.
这意味着内积$\left\langle \cdot, \cdot \right\rangle$被矩阵$A$*唯一确定*，

The symmetry of the inner product also means that A is symmetric.
且由于内积具有对称性，不难看出$A$是对称矩阵。

Furthermore, the positive definiteness of the inner product implies that
进一步地，根据内积的正定性，我们可以得出下面的结论：
$$
\forall x \in V - \{ 0 \}: x^{\top}Ax > 0. \tag{3.11}
$$

![500](Pasted%20image%2020240302125526.png)

> **定义 3.4**（对称正定矩阵）
> 一个$n$级对称矩阵$A \in \mathbb{R}^{n \times n}$若满足（式 3.11），则叫做*对称正定矩阵*（或仅称为正定矩阵）。如果只满足将（式 3.11）中的不等号改成$\geqslant$的条件，则称为*对称半正定矩阵*

![500](Pasted%20image%2020240302125539.png)

> **示例 3.4**（对称正定矩阵）
> 考虑下面两个矩阵
> $$A_{1} = \left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right] , \quad A_{2} = \left[  \begin{matrix} 9 & 6 \\ 6 & 3 \end{matrix} \right], \tag{3.12}$$
> 其中 $A_{1}$ 是对称且正定的，因为它不仅对称（译者注：这显而易见），而且对于任意 $x \in \mathbb{R}^{2} - \{ 0 \}$ 都有，
> $$\begin{align} x^{\top}A_{1}x &= \left[ \begin{matrix} x_{1} & x_{2} \end{matrix}\right]\left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right]\left[ \begin{matrix} x_{1} \\ x_{2}  \end{matrix}\right] \\\ &= 9x_{1}^{2} + 12x_{1}x_{2} + 5x_{2}^{2} \\ &= (3x_{1} + 2x_{2})^{2} + x_{2}^{2} > 0.\end{align} \tag{3.13}$$
> 相反地，$A_{2}$不是正定矩阵。如果取$x = [2, -3]^{\top}$，可以验证二次型$x^\top Ax$是负数。

If A ∈ Rn×n is symmetric, positive definite, then [formula 3.15] defines an inner product with respect to an ordered basis B, where ˆx and
ˆy are the coordinate representations of x, y ∈ V with respect to B.

假设$A \in \mathbb{R}^{n \times n}$是一个对称正定矩阵，则它可以定义一个在基$B$下的内积：

$$
\left\langle x, y \right\rangle = \hat{x}^{\top}A\hat{y}, \tag{3.15}
$$

其中$x, y \in V$。

![500](Pasted%20image%2020240302131154.png)

> **定理 3.5**
> 考虑一个有限维实向量空间$V$及它的一个基（有序）$B$，双线性函数$\left\langle \cdot, \cdot \right\rangle: V \times V \rightarrow R$是其上的一个内积<u>当且仅当</u>有一个对称正定矩阵$A \in \mathbb{R}^{n \times n}$，与之对应，即
> $$\left\langle x, y \right\rangle = \hat{x}^{\top} A \hat{y}.$$

The following properties hold if A ∈ Rn×n is symmetric and positive definite:
下面再列举两个对称正定矩阵的性质

* The null space (kernel) of A consists only of 0 because x Ax > 0 for all x = 0. This implies that Ax = 0 if x = 0.
* 矩阵$A$的零空间（核）只包含零向量，因为若$x$不为零，则有$x^{\top}Ax>0$，于是$Ax \ne 0$。

* The diagonal elements aii of A are positive because aii = e i Aei > 0, where ei is the ith vector of the standard basis in Rn
* 矩阵$A$的所有对角元（$a_{ii}$，$i=1, \dots, n$）都是正数，因为$a_{i i} = e_{i}^{\top} Ae_{i} > 0$，其中$e$是$B$中第$i$个基向量。

## 3.3 向量长度和距离（75）

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

Remark (Cauchy-Schwarz Inequality). For an inner product vector space (V, h·, ·i) the induced norm k · k satisfies the Cauchy-Schwarz inequality
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
> $$\begin{align} d: V \times V & \rightarrow \mathbb{R} \tag{3.22}\\ (x, y) & \mapsto d(x, y) \tag{3.23}\end{align}$$
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

## 3.4 向量夹角和正交（76）

In addition to enabling the definition of lengths of vectors, as well as the distance between two vectors, inner products also capture the geometry of a vector space by defining the angle ω between two vectors. We use the Cauchy-Schwarz inequality (3.17) to define angles ω in inner product spaces between two vectors x, y, and this notion coincides with our intuition in R2 and R3. 
Assume that x = 0, y = 0. Then
在对向量的长度和两向量之间的距离进行定义的基础上，内积还可以通过定义两向量之间的夹角$\omega$以刻画向量空间中的几何特征。我们使用Cauchy-Schwarz不等式（3.17）定义内积空间中两个向量$x$和$y$之间的夹角$\omega$，这和我们在$\mathbb{R}^{2}$和$\mathbb{R}^{3}$中的结论相同。假设两个向量均布为零，我们有
$$
-1 \leqslant \frac{\left\langle x, y \right\rangle}{\|x\| \|y\|} \leqslant 1. \tag{3.24}
$$

Therefore, there exists a unique ω ∈ [0, π], illustrated in Figure 3.4, with
如图3.4所示，在$[0, \pi]$中有唯一的$\omega$满足下面的等式：
$$
\cos\omega = \frac{\left\langle x, y \right\rangle}{\|x\| \|y\|}. \tag{3.25}
$$

![300](Pasted%20image%2020240627164202.png)

Figure 3.4 When restricted to [0, π] then f(ω) = cos(ω) returns a unique number in the interval [−1, 1].
图3.4 定义域为$[0, \pi]$时的余弦函数图像，此时角度值和余弦函数值一一对应

The number ω is the angle between the vectors x and y. Intuitively, the angle angle between two vectors tells us how similar their orientations are. For example, using the dot product, the angle between x and y = 4x, i.e., y is a scaled version of x, is 0: Their orientation is the same.
而$\omega$就是$x$和$y$之间的夹角。直观意义上，两向量之间的夹角给出了其方向的相似程度，例如两向量$x$和$y=4x$（$x$经过常数缩放后的版本）的夹角为零，因此它们的方向相同。

> Example 3.6 (Angle between Vectors)
> Let us compute the angle between x = [1, 1]> ∈ R2 and y = [1, 2]> ∈ R2 ; see Figure 3.5, where we use the dot product as the inner product. Then we get
> **示例 3.6 （向量之间的夹角）**
> 如图3.5所示，计算向量$x = [1, 1]^{\top} \in \mathbb{R}^{2}$和$y = [1, 2]^{\top} \in \mathbb{R}^{2}$的夹角。我们令向量的内积为点积，有
> $$ \cos\omega = \frac{\left\langle x, y \right\rangle}{\sqrt{ \left\langle x, x \right\rangle \left\langle y, y \right\rangle  }} = \frac{x^{\top}y}{\sqrt{ x^{\top}xy^{\top}y }} = \frac{3}{\sqrt{ 10 }}, \tag{3.26} $$
> and the angle between the two vectors is arccos( √ 3  10 ) ≈ 0.32 rad, which corresponds to about 18◦
> 于是两个向量的夹角余弦值为$\displaystyle \arccos\left( \frac{3}{\sqrt{ 10 }} \right) \approx 0.32\text{ rad}$，大约为$18^{\circ}$。
>  

![150](Pasted%20image%2020240627165050.png)

> Figure 3.5 The angle ω between two vectors x, y is computed using the inner product.
> 图3.5 使用向量$x$和$y$之间的内积计算它们之间的夹角$\omega$。

A key feature of the inner product is that it also allows us to characterize vectors that are orthogonal.
内积的一个关键用途是判断向量之间是否正交。

> Definition 3.7 (Orthogonality). 
> Two vectors x and y are orthogonal if and orthogonal only if h x, yi = 0, and we write x ⊥ y. If additionally k xk = 1 = k yk , i.e., the vectors are unit vectors, then x and y are orthonormal. orthonormal An implication of this definition is that the 0-vector is orthogonal to every vector in the vector space.
> **定义3.7（向量的正交）**
> 两个向量$x$和$y$ *正交（orthogonal）* 当且仅当它们的内积为零，即$\left\langle x, y \right\rangle = 0$。
> 进一步地，如果$\|x\| = \|y\| = 1$，也即两个向量是单位向量，则称它们 *单位正交（orthonormal）* 。
> 特殊地，零向量$\boldsymbol{0}$与任意向量都正交。

Remark. Orthogonality is the generalization of the concept of perpendicularity to bilinear forms that do not have to be the dot product. In our context, geometrically, we can think of orthogonal vectors as having a right angle with respect to a specific inner product. ♦
注：正交性将垂直这一概念推广至通常点积之外的双线性型范畴。在我们的讨论中，可以从几何的角度认为在某一内积下正交的两个向量的夹角为直角。

> Example 3.7 (Orthogonal Vectors)
> Figure 3.6 The  angle ω between two vectors x, y can change depending on the inner product.
> Consider two vectors x = [1, 1]> , y = [−1, 1]> ∈ R2  ; see Figure 3.6. We are interested in determining the angle ω between them using two different inner products. Using the dot product as the inner product yields an angle ω between x and y of 90◦ , such that x ⊥ y. However, if we choose the inner product
> **示例3.7（单位正交向量）**
>  

![300](Pasted%20image%2020240627170238.png)

> 图3.6 使用不同的内积定义计算得到的两向量$x$和$y$之间的夹角不同
> 如图3.6所示，考虑向量$x=[1, 1]^{\top}, y = [-1, 1]^{\top} \in \mathbb{R}^{2}$，考虑它们在不同内积定义下的夹角大小。如果使用通常的点积作为内积，则它们之间的夹角为$90^{\circ}$，也即$x \bot y$。但如果使用下面的内积定义则会得到不同的结果： $$ \left\langle x, y \right\rangle  = x^{\top} \left[ \begin{matrix} 2 & 0 \\ 0 & 1

\end{matrix}\right] y, \tag{3.27} $$

> we get that the angle ω between x and y is given by
> 可以计算得到在这个内积之下两向量之间的夹角为> $$ \cos \omega = \frac{\left\langle x, y \right\rangle}{\|x\| \|y\|} = -\frac{1}{3} \implies \omega \approx 1.91 \text{ rad} \approx 109.5^{\circ}, \tag{3.28} $$
> and x and y are not orthogonal. Therefore, vectors that are orthogonal with respect to one inner product do not have to be orthogonal with respect to a different inner product.
> 于是$x$和$y$并不正交。因此在一个内积下正交的两个向量在另一个内积下不一定正交。

> Definition 3.8 (Orthogonal Matrix). 
> A square matrix A ∈ Rn×n is an orthogonal matrix if and only if its columns are orthonormal so that
> **定义3.8（正交矩阵）**
> 方阵$A \in \mathbb{R}^{n \times n}$为正交矩阵当且仅当满足下面的条件：
> $$ A A^{\top} = I = A^{\top} A, \tag{3.29} $$
> which implies that
> 进而有
> $$ A^{-1} = A^{\top}, \tag{3.30} $$
> i.e., the inverse is obtained by simply transposing the matrix. 
> 这是说，正交矩阵的逆是它的转置。

Remark：It is convention to call these matrices “orthogonal” but a more precise description would be “orthonormal”. Transformations with orthogonal matrices preserve distances and angles.
注：一般我们将这些矩阵称为“orthogonal matrix”，严格意义上它们应该叫做“orthonormal matrix”。因为“orthonormal matrix”对应的变换在线性空间内保持向量的长度和向量之间的夹角。译者注：中文中不做区分，统一称为“正交矩阵”。正交矩阵对应的变换在$\mathbb{R}^{2}$和$\mathbb{R}^{3}$中属于刚体变换。

Transformations by orthogonal matrices are special because the length of a vector x is not changed when transforming it using an orthogonal matrix A. For the dot product, we obtain
正交矩阵作用在向量$x$上不改变它的长度。当范数为向量点积作为内积诱导的范数时，我们有
$$
\|Ax\|^{2} = (Ax)^{\top}(Ax) = x^{\top}A^{\top}Ax = x^{\top}I x = x^{\top}x = \|x\|^{2}. \tag{3.31}
$$
Moreover, the angle between any two vectors x, y, as measured by their inner product, is also unchanged when transforming both of them using an orthogonal matrix A. Assuming the dot product as the inner product, the angle of the images Ax and Ay is given as
进一步地，两个向量$x$和$y$之间使用内积度量的夹角在同时被正交矩阵作用后依然保持不变。假设内积依然为点积，$Ax$和$Ay$之间的夹角余弦值为
$$
\cos \omega = \frac{(Ax)^{\top}(Ay)}{\|Ax\|\|Ay\|} = \frac{x^{\top}A^{\top}Ay}{\sqrt{ x^{\top}A^{\top}Axy^{\top}A^{\top}Ay }} = \frac{x^{\top}y}{\|x\| \|y\|}, \tag{3.32}
$$
which gives exactly the angle between x and y. This means that orthogonal matrices A with A  = A −1 preserve both angles and distances. It turns out that orthogonal matrices define transformations that are rotations (with the possibility of flips). In Section 3.9, we will discuss more details about rotations.
以上内容表示，正交矩阵对应的线性变换同时保持长度和夹角。事实上，这些正交矩阵定义了一系列的旋转和翻转。在章节3.9中我们会进一步讨论它们。

## 3.5 正交基

In Section 2.6.1, we characterized properties of basis vectors and found
that in an n-dimensional vector space, we need n basis vectors, i.e., n
vectors that are linearly independent. In Sections 3.3 and 3.4, we used
inner products to compute the length of vectors and the angle between
vectors. In the following, we will discuss the special case where the basis
vectors are orthogonal to each other and where the length of each basis
vector is 1. We will call this basis then an orthonormal basis.

在2.6.1节中，我们讨论了基向量的性质，我们发现在$n$维空间中，我们需要$n$个基向量（也就是$n$个线性无关的向量）。在3.3和3.4两节中，我们使用内积计算向量的长度和向量之间的夹角。在本节中，我们将讨论基向量互相垂直且长度为$1$这一特殊情况，我们称其为**正交基**。

Let us introduce this more formally.
Definition 3.9 (Orthonormal Basis). Consider an n-dimensional vector
space V and a basis {b1, ... , bn} of V . If for all i, j = 1, ... , n then the basis is called an orthonormal basis (ONB). orthonormal basis
ONB If only (3.33) is satisfied, then the basis is called an orthogonal basis. Note
orthogonal basis that (3.34) implies that every basis vector has length/norm 1.

我们不妨使用更加严谨的语言介绍它们：

> **定义 3.9 （正交基）**
> 考虑一个$n$维向量空间$V$和它上面的一个基$\{ b_{1}, \dots, b_{n} \}$，如果$$\begin{align}\left\langle b_{i}, b_{j} \right\rangle &= 0, & i \ne j \tag{3.33}\\\left\langle b_{i}, b_{i} \right\rangle &= 1 \tag{3.34}\end{align}$$
> 对于所有的$i, j = 1, \dots, n$都成立，那么$\{ b_{1}, \dots, b_{n} \}$就叫做**标准正交基（orthonormal basis，ONB）**，注意所有的向量的长度均为$1$。假如这个基只满足$(3.33)$，则它就叫做**正交基（orthogonal basis）**。

Recall from Section 2.6.1 that we can use Gaussian elimination to find a
basis for a vector space spanned by a set of vectors. Assume we are given
a set {
˜b1, ... , 
˜bn} of non-orthogonal and unnormalized basis vectors. We
concatenate them into a matrix ˜B = [˜b1, ... , 
˜bn] and apply Gaussian elimination to the augmented matrix (Section 2.3.2) [
˜B ˜B
 |
˜B] to obtain an
orthonormal basis. This constructive way to iteratively build an orthonormal basis {b1, ... , bn} is called the Gram-Schmidt process (Strang, 2003).

让我们回忆一下，在2.6.1节中我们使用Gauss消元法寻找一个向量组张成空间的基的过程。假设我们有一个未标准化（unnormalized）且非正交的向量组$\{ \tilde{b}_{1}, \dots, \tilde{b}_{n} \}$，我们将其堆叠成一个矩阵$\tilde{B} = [\tilde{b}_{1}, \dots, \tilde{b}_{n}]$，然后在增广矩阵$[\tilde{B}\tilde{B}^{\top}|\tilde{B}]$上应用Gauss消元法，就可以得到一个标准正交基。像这样迭代地构造正交基$\{ b_{1}, \dots, b_{n} \}$的方法叫做**Gram-Schmidt正交化过程**。

Example 3.8 (Orthonormal Basis)
The canonical/standard basis for a Euclidean vector space Rn
is an orthonormal basis, where the inner product is the dot product of vectors.
In R2
, the vectors form an orthonormal basis since b1 b2 = 0 and k b1k = 1 = k b2k .
We will exploit the concept of an orthonormal basis in Chapter 12 and
Chapter 10 when we discuss support vector machines and principal component analysis.

> **示例 3.8（正交基）**
> Euclid空间$\mathbb{R}^{n}$上的标准基是标准正交基，其中内积为两个向量的点积。
> 特别地，在$\mathbb{R}^{2}$中，两个向量$$b_{1} = \frac{1}{\sqrt{ 2 }} \left[ \begin{matrix}1\\1\end{matrix} \right], \quad b_{2} = \frac{1}{\sqrt{ 2 }} \left[ \begin{matrix} 1\\-1 \end{matrix} \right] \tag{3.35}$$组成一个正交基，因为$b_{1}^{\top}b_{2} =0$且$\|b_{1}\| = \|b_{2}\| = 1$。

We will exploit the concept of an orthonormal basis in Chapter 12 and
Chapter 10 when we discuss support vector machines and principal component analysis.

我们将在第十章和第十二章介绍支持向量机和主成分分析时深入讲解标准正交基这一概念。

## 3.6 正交补

Having defined orthogonality, we will now look at vector spaces that are
orthogonal to each other. This will play an important role in Chapter 10, 
when we discuss linear dimensionality reduction from a geometric perspective.

我们在定义了正交这一概念之后可以来看看互相正交的向量空间了。这样的向量空间在第十章讨论线性降维的几何视角时十分重要。

Consider a D-dimensional vector space V and an M-dimensional subspace U ⊆ V . Then its orthogonal complement U⊥ is a (D−M)-dimensional orthogonal complement subspace of V and contains all vectors in V that are orthogonal to every vector in U. Furthermore, U ∩ U ⊥ = {0} so that any vector x ∈ V can be uniquely decomposed into

考虑一个$D$维的向量空间$V$和一个$M$维的子空间$U \subset V$，$U$的**正交补（orthogonal complement）**$U^{\perp}$是一个$(D-M)$维的子空间，其中的任何向量都与$U$中的任何向量垂直。进一步我们有$U \cap U^{\perp} = \{ 0 \}$，于是$V$中的任何向量$x$可以被唯一分解为下面的形式：
$$
x = \sum\limits_{m=1}^{M} \lambda b_{m} + \sum\limits_{j=1}^{D-M} \psi_{i}b_{j}^{\bot}, \quad \lambda_{m}, \psi_{j} \in \mathbb{R} \tag{3.36}
$$
where (b1, ... , bM) is a basis of U and (b⊥1, ... , b⊥D−M) is a basis of U⊥.

其中$(b_{1}, \dots, b_{M})$是$U$的一个基，$(b^{\perp}_{1}, \dots, b^{\perp}_{D-M})$是$U^{\perp}$的一个基。

![300](Pasted%20image%2020240809151317.png)

<center>图3.7 三维向量空间的平面可被与其垂直单位向量唯一确定，后者是其正交补空间的基</center>

Therefore, the orthogonal complement can also be used to describe a
plane U (two-dimensional subspace) in a three-dimensional vector space.
More specifically, the vector w with k wk = 1, which is orthogonal to the
plane U, is the basis vector of U⊥. Figure 3.7 illustrates this setting. All vectors that are orthogonal to w must (by construction) lie in the plane
U. The vector w is called the normal vector of U.

因此，三维向量空间中的平面$U$的正交补也可以用来描述平面本身（平面是二维的）。具体来说，三维空间的一个向量$w$如果满足$\|w\| = 1$，是某个平面$U$的正交补空间$U^{\perp}$的一个基，如图3.7所示。在图中，所有与$w$垂直的向量一定在平面$U$中，故$w$也被称作平面$U$的**法向量（normal vector）**

Generally, orthogonal complements can be used to describe hyperplanes
in n-dimensional vector and affine spaces.
一般地，正交补空间可被用来刻画$n$维向量空间和仿射空间（affine space，译者注：也称线性流形）中的**超平面（hyperplanes）**

## 3.7 函数的内积

Thus far, we looked at properties of inner products to compute lengths, angles and distances. We focused on inner products of finite-dimensional vectors. In the following, we will look at an example of inner products of a different type of vectors: inner products of functions.
到现在我们了解了内积的各种性质，并利用它们计算有限维向量的长度、夹角和距离。在本节中，我们将看到另一种向量之间的内积：函数的内积。

The inner products we discussed so far were defined for vectors with a finite number of entries. We can think of a vector x ∈ Rn as a function with n function values. The concept of an inner product can be generalized to vectors with an infinite number of entries (countably infinite) and also continuous-valued functions (uncountably infinite). Then the sum over individual components of vectors (see Equation (3.5) for example) turns into an integral.
到此为止我们讨论的所有内积都定义在具有有限个分量的向量之上。我们可以将向量$x \in \mathbb{R}^{n}$视作有$n$个取值的函数，这样一来内积的概念可以推广至具有无限个分量（可数无穷）以及连续（不可数无穷）的向量之上。在这样的意义下，原来对不同向量分量的（乘积后）的加和（例如式$(3.5)$）将变为积分。

An inner product of two functions u : R → R and v : R → R can be defined as the definite integral
两个函数$u: \mathbb{R} \rightarrow \mathbb{R}$和$v: \mathbb{R} \rightarrow \mathbb{R}$之间的内积可被定义为下面的定积分：
$$
\left\langle u, v \right\rangle := \int_{a}^{b} {u(x)v(x)} \, \mathrm d{x}, \tag{3.37} 
$$
for lower and upper limits a, b < ∞, respectively.
其中积分限满足$a, b < \infty$。

As with our usual inner product, we can define norms and orthogonality by looking at the inner product. If (3.37) evaluates to 0, the functions u and v are orthogonal. To make the preceding inner product mathematically precise, we need to take care of measures and the definition of integrals, leading to the definition of a Hilbert space. Furthermore, unlike inner products on finite-dimensional vectors, inner products on functions may diverge (have infinite value). All this requires diving into some more intricate details of real and functional analysis, which we do not cover in this book.
和通常的内积一样，我们也可以通过内积定义函数的范数和正交关系。如果式$(3.37)$的结果为零，则两个函数$u$和$v$相互正交。如果需要给出更加严格的定义，我们需要考虑测度和积分定义的方式，这将引出Hilbert空间。进一步地，与有限维向量之间的内积不同，函数之间的内积可能发散（值为无穷大）。对上述情形的讨论涉及实分析和泛函分析中的细节，不是本书讨论的内容。

Example 3.9 (Inner Product of Functions)
If we choose u = sin(x) and v = cos(x), the integrand f(x) = u(x)v(x) Figure 3.8 f(x) =
sin(x) cos(x).
of (3.37), is shown in Figure 3.8. We see that this function is odd, i.e., 
f(−x) = −f(x). Therefore, the integral with limits a = −π, b = π of this
product evaluates to 0. Therefore, sin and cos are orthogonal functions.

> **示例 3.9（函数之间的内积）**
> 假如我们令$u = \sin(x)$，$v = \cos(x)$，则内积定义$(3.37)$中的被积函数为$f = u(x)v(x)$，如图3.38所示。我们发现这个函数是奇函数，也即$f(-x) = -f(x)$。所以积分限为$a=-\pi, b=\pi$的定积分的值为零，因此我们可以得到$\sin$和$\cos$互相正交的结论。
>  

![300](Pasted%20image%2020240811182908.png)

> <center>图3.8 被积函数 f(x) = sin(x)cos(x) 的图像</center>
>  

Remark. It also holds that the collection of functions is orthogonal if we integrate from −π to π, i.e., any pair of functions are orthogonal to each other. The collection of functions in (3.38) spans a large subspace of the functions that are even and periodic on \[−π, π\), and projecting functions onto this subspace is the fundamental idea behind
Fourier series.

> 注：上述结论对于下面的函数族依然成立：$$\{ 1, \cos(x), \cos(2x), \cos(3x), \dots \}, \tag{3.38}$$（如果将积分限设置为$-\pi$和$\pi$）。换句话说，这个函数族中的函数两两正交，它们张成的巨大空间是所有以区间$[-\pi, \pi)$为周期的连续函数。将函数向这个子空间上投影是**Fourier级数**的核心思想。

In Section 6.4.6, we will have a look at a second type of unconventional
inner products: the inner product of random variables.
在6.4.6节，我们还会遇见第二种不常见的内积——随机变量之间的内积。

## 3.8 正交投影

Projections are an important class of linear transformations (besides rotations and reflections) and play an important role in graphics, coding theory, statistics and machine learning. 
投影是一类重要的线性变换（其他重要的线性变换还有旋转和反射），在图形学、编码理论、统计学和机器学习中占有重要的地位。

In machine learning, we often deal with data that is high-dimensional. High-dimensional data is often hard to analyze or visualize. 
在机器学习中，我们经常需要与高维数据打交道，它们往往难以进行分析和可视化。

However, high-dimensional data quite often possesses the property that only a few dimensions contain most information, and most other dimensions are not essential to describe key properties of the data. 
然而，高维数据往往具有大部分信息被包含在仅仅几个维度之中，其他维度对于数据关键信息的刻画并不重要的特点。

When we compress or visualize high-dimensional data, we will lose information. To minimize this compression loss, we ideally find the most informative dimensions in the data. 
当我们对高维数据进行压缩或可视化时，我们将失去一些信息。为了将压缩造成的信息损失最小化，我们往往选择数据中最关键的几个维度。

As discussed in Chapter 1, data can be represented as vectors, and in this chapter, we will discuss some of the fundamental tools for data compression. 
我们在第一章中提到，数据可被表示成向量。在本章中，我们将对基础的数据压缩方法进行讨论。

More specifically, we can project the original high-dimensional data onto a lower-dimensional feature space and work in this lower-dimensional space to learn more about the dataset and extract relevant patterns. 
具体而言，我们可以将原来的高维数据投影到低维**特征空间（feature space）**，然后在此空间中对数据进行处理和分析，以更好的了解数据集并抽取相关的**模式（pattern）**。

> “Feature” is a common expression for data representation.
> “特征”是数据表示的一个常见说法。

For example, machine learning algorithms, such as principal component analysis (PCA) by Pearson (1901) and Hotelling (1933) and deep neural networks (e.g., deep auto-encoders (Deng et al., 2010)), heavily exploit the idea of dimensionality reduction. 
举例来说，以主成分分析（principal component analysis，PCA）为例的机器学习算法（Pearson, 1901 和 Hotelling, 1933）以及以自编码器（auto-encoders，Deng et al., 2010）深度神经网络充分利用了降维的思想。

In the following, we will focus on orthogonal projections, which we will use in Chapter 10 for linear dimensionality reduction and in Chapter 12 for classification. 
接下来，我们将将注意力集中在第十章将被使用于线性降维和在十二章中的分类问题中的正交投影上。

Even linear regression, which we discuss in Chapter 9, can be interpreted using orthogonal projections. 
即使是我们将在第九章中讨论的线性回归算法，也可以从正交投影的角度进行解读。

For a given lower-dimensional subspace, orthogonal projections of high-dimensional data retain as much information as possible and minimize the difference/error between the original data and the corresponding projection. 
给定一个低维子空间，来自高维空间中数据的正交投影会保留尽可能多的信息，并最小化元数据和投影数据的区别或损失。

![400](Pasted%20image%2020240813213937.png)
<center>图 3.9 二维数据点（蓝色点）至一维子空间（直线）的投影（橙色点）</center>

An illustration of such an orthogonal projection is given in Figure 3.9. Before we detail how to obtain these projections, let us define what a projection actually is.
正交投影的直观几何描述可见图 3.9。在我们介紹细节之前，需要首先定义投影这个概念。

> **定义 3.10 （投影）**
> 令 $V$ 为一个向量空间，$U\subset V$ 是 $V$ 的子空间，如果一个线性映射 $\pi: V \rightarrow U$ 满足 $\pi^2 = \pi \circ \pi = \pi$，则称 $\pi$ 为一个**投影（projection）**。

Since linear mappings can be expressed by transformation matrices (see Section 2.7), the preceding definition applies equally to a special kind
of transformation matrices, the projection matrices P π, which exhibit the property that P2 π = P π.
由于线性映射可以表示为矩阵（参见 2.7 节），上面的定义等价于确定了一类特殊的矩阵变换 $P_\pi$，它们满足 $P_\pi^2 = P_\pi$。

In the following, we will derive orthogonal projections of vectors in the inner product space (Rn , h·, ·i) onto subspaces. We will start with one dimensional subspaces, which are also called lines. If not mentioned otherwise, we assume the dot product h x, yi = xTy as the inner product.
在接下来的内容中将推导内积空间 $(\mathbb{R}^n, \langle \cdot, \cdot \rangle)$ 中向量至其子空间的正交投影，我们将从一维子空间（也称为直线）开始。如果没有特殊说明，我们约定向量的内积为点积，即 $\langle x, y \rangle = x^\top y$。

### 3.8.1 向一维子空间（直线）投影

Assume we are given a line (one-dimensional subspace) through the origin with basis vector b ∈ Rn
. The line is a one-dimensional subspace U ⊆ Rn spanned by b. When we project x ∈ Rn onto U, we seek the vector πU (x) ∈ U that is closest to x. Using geometric arguments, let us characterize some properties of the projection πU (x) (Figure 3.10(a)
serves as an illustration):
假设给定一条通过原点的直线（一维子空间），和该空间的一个基 $b \in \mathbb{R}^n$。这条直线是$b$章程的子空间$U \subset \mathbb{R}^n$。当我们将向量$x \in \mathbb{R}^n$投影至$U$中时，我们需要在$U$中寻找距离$x$最近的向量$\pi_U(x) \in U$。下面列举一些投影向量$\pi_U(x)$的性质（参考图 3.10）

<!-- TODO: 需要将其中的英文改成中文 -->

* The projection πU (x) is closest to x, where “closest” implies that the distance k x−πU (x)k is minimal. It follows that the segment πU (x)−x from πU (x) to x is orthogonal to U, and therefore the basis vector b of U. The orthogonality condition yields h πU (x) − x, bi = 0 since angles between vectors are defined via the inner product.
* 投影向量 $\pi_U(x)$ 是（子空间中）距离 $x$ 最近的向量，“最近”的意思是距离 $\|x - \pi_U(x)\|$ 是最小的。这表示从 $\pi_U(x)$ 到 $x$ 的线段 $\pi_U(x) - x$ 与 $U$ 是垂直的，也和 $U$ 的基 $b$ 垂直。

* The projection πU (x) of x onto U must be an element of U and, therefore, a multiple of the basis vector b that spans U. Hence, πU (x) = λb, for some λ ∈ R.
* $x$ 到 $U$ 的投影向量 $\pi_U(x)$ 一定是 $U$ 中的元素，因此也和 $U$ 的基 $b$ 共线。于是存在 $\lambda \in \mathbb{R}$，使得 $\pi_U(x) = \lambda b$。

> Remark. λ is then the coordinate of πU (x) with respect to b.
> 注：$\lambda$ 是 $\pi_{U}(\boldsymbol{x})$ 在基 $b$ 下的坐标。

In the following three steps, we determine the coordinate λ, the projection
πU (x) ∈ U, and the projection matrix P π that maps any x ∈ Rn onto U:
下面我们将通过三个步骤确定坐标 $\lambda$，投影向量 $\pi_{U}(\boldsymbol{x}) \in U$，以及将 $x \in \mathbb{R}^{n}$ 投影至子空间 $U$ 的投影矩阵 $\boldsymbol{P}_{\pi}$ 。

1. Finding the coordinate λ. The orthogonality condition yields
    计算坐标 $\lambda$ 的值。由正交性条件得到
    $$
    \left\langle x - \pi_{U}(\boldsymbol{x}), b \right\rangle = 0 \mathop{\iff}\limits^{\pi_{U}(\boldsymbol{x}) = \lambda b} \left\langle x - \lambda b, b \right\rangle = 0. \tag{3.39}  
    $$
    
    We can now exploit the bilinearity of the inner product and arrive at
    我们可以利用内积的双线性性，得到
    $$
    \left\langle x, b \right\rangle - \lambda\left\langle b, b \right\rangle = 0 \iff \lambda = \frac{\left\langle x, b \right\rangle}{\left\langle b, b \right\rangle } = \frac{\left\langle b, x \right\rangle}{\|b\|^{2}} . \tag{3.40}
    $$

    > With a general inner product, we get λ = h x, bi if k bk = 1.
    > 若使用一般的内积，如果$\|b\| = 1$，我们有 $\lambda = \left\langle x, b \right\rangle$。

    In the last step, we exploited the fact that inner products are symmetric. If we choose h·, ·i to be the dot product, we obtain
    最后，我们利用内积的对称性对原式进行变换。如果我们令 $\left\langle \cdot, \cdot \right\rangle$ 为点积，我们就可以得到
    $$
    \lambda = \frac{b^{\top}x}{b^{\top}b} = \frac{b^{\top}x}{\|b\|^{2}}. \tag{3.41}
    $$

    If k bk = 1, then the coordinate λ of the projection is given by bTx.
    如果 $\|b\| =1$，则 $\lambda$ 的值为 $b^{\top}x$。

2. Finding the projection point πU (x) ∈ U. Since πU (x) = λb, we immediately obtain with (3.40) that 
   计算投影点 $\pi_{U}(\boldsymbol{x}) \in U$。由于 $\pi_{U}(\boldsymbol{x}) = \lambda b$，由 $(3.40)$，立刻有
    $$
    \pi_{U}(\boldsymbol{x}) = \lambda b = \frac{\left\langle x, b \right\rangle}{\|b\|^{2}} \cdot b = \frac{b^{\top}x}{\|b\|^{2}} \cdot b, \tag{3.42}
    $$

    where the last equality holds for the dot product only. We can also compute the length of πU (x) by means of Definition 3.1 as
    其中最后的等号成立条件为内积取为点积。我们还可以根据定义3.1计算 $\pi_U(x)$ 的长度：
    $$
    \|\pi_{U}(\boldsymbol{x})\| = \|\lambda b\| = |\lambda| \|b\|. \tag{3.43}
    $$

    Hence, our projection is of length |λ| times the length of b. This also adds the intuition that λ is the coordinate of πU (x) with respect to the basis vector b that spans our one-dimensional subspace U.
    因此，投影向量的长度为 $|\lambda|$ 乘以 $b$ 的长度。这也增加了一个直观理解方式：$\lambda$ 是投影向量在子空间 $U$ 的基 $b$ 下的坐标。
    
    If we use the dot product as an inner product, we get
    如果我们令内积为点积，就有
    $$
    \begin{align}
    \|\pi_{U}(\boldsymbol{x})\| ~&\mathop{=\!=\!=}\limits^{(3.42)} ~\frac{|b^{\top}x|}{\|b\|^{2}} \|b\|~ \mathop{=\!=\!=}\limits^{(3.25)} ~|\!\cos\omega| \cdot \|x\| \cdot \|b\| \cdot \frac{\|b\|}{\|b\|^{2}} \\&= |\!\cos{\omega}| \cdot \|x\|.
    \end{align} \tag{3.44}
    $$

    
![600](Pasted%20image%2020240813214150.png)
<center>图 3.10 投影至一位子空间的示例。</center>

Here, ω is the angle between x and b. This equation should be familiar from trigonometry: If k xk = 1, then x lies on the unit circle. It follows that the projection onto the horizontal axis spanned by b is exactly cos ω, and the length of the corresponding vector πU (x) = |cos ω|. An
    illustration is given in Figure 3.10(b).
    这里的 $\omega$ 是向量 $x$ 和$b$ 之间的夹角。如图3.10所示，从三角学的角度看，该结果是似曾相识的：如果 $\|x\| = 1$，则向量 $x$ 的终点位于单位圆上。接着可以得到 $x$ 向横轴的投影在基 $b$ 下的坐标恰好就是 $\cos \omega$，投影向量的长度也满足 $|\pi_{U}(\boldsymbol{x})| = |\cos\omega|$。

> The horizontal axis is a one-dimensional subspace.
    > 注：所谓的横轴就是一个一维子空间。


3. Finding the projection matrix P π. We know that a projection is a linear mapping (see Definition 3.10). Therefore, there exists a projection matrix P π, such that πU (x) = P πx. With the dot product as inner product and
    计算投影矩阵 $\boldsymbol{P}_{\pi}$。通过定义 3.10 我们知道投影是一个线性变换。因此存在一个投影矩阵$\boldsymbol{P}_{\pi}$，使得 $\pi_{U}(\boldsymbol{x}) = \boldsymbol{P}_{\pi} x$。若令点积为内积，我们有
    $$
    \pi_{U}(\boldsymbol{x}) = \lambda b = b\lambda =b \frac{b^{\top}x}{\|b\|^{2}} = \frac{bb^{\top}}{\|b\|^{2}} x, \tag{3.45}
    $$

    we immediately see that
    这样立刻得到
    $$
    \boldsymbol{P}_{\pi} = \frac{b b^{\top}}{\|b\|^{2}}. \tag{3.46}
    $$

    Note that bb> (and, consequently, P π) is a symmetric matrix (of rank 1), and kbk2 = h b, bi is a scalar.
    注意 $bb^{\top}$（也就是 $\boldsymbol{P}_{\pi}$）是秩为 $1$ 的对称矩阵，而 $\|b\|^{2} = \left\langle b, b \right\rangle$ 是一个标量。

The projection matrix P π projects any vector x ∈ Rn onto the line through the origin with direction b (equivalently, the subspace U spanned by b).
投影矩阵 $\boldsymbol{P}_{\pi}$ 将任意向量 $x \in \mathbb{R}^{n}$ 投影到通过原点，方向为 $b$ 的直线上（这等价于由 $b$ 张成的子空间 $U$）。

> Remark. The projection πU (x) ∈ Rn is still an n-dimensional vector and not a scalar. However, we no longer require n coordinates to represent the projection, but only a single one if we want to express it with respect to the basis vector b that spans the subspace U: λ. ♦
> 注：投影向量 $\pi_{U}(\boldsymbol{x}) \in \mathbb{R}^{n}$ 依然是一个 $n$ 维向量，不是一个标量。然而，我们不再需要使用 $n$ 个分量来描述它——我们只需要使用一个分量 $\lambda$，因为这是投影向量关于子空间 $U$ 中的基 $b$ 的坐标。

Example 3.10 (Projection onto a Line)
Find the projection matrix P π onto the line through the origin spanned by b = 1 2 2 > . b is a direction and a basis of the one-dimensional subspace (line through origin). With (3.46), we obtain
Let us now choose a particular x and see whether it lies in the subspace
spanned by b. For x =  1 1 1 > , the projection is
Note that the application of P π to πU (x) does not change anything, i.e., P ππU (x) = πU (x). This is expected because according to Definition 3.10, we know that a projection matrix P π satisfies P2π x = Pπ x for all x

> **示例 3.10（向直线投影）**
> 求投影至通过原点，由向量 $b = [1, 2, 2]^{\top}$ 张成直线的投影矩阵 $\boldsymbol{P}_{\pi}$，其中 $b$ 是该过原点直线的方向，也就是一维子空间的基。
> 
> 通过 $(3.46)$，我们有
> $$\boldsymbol{P}_{\pi} = \frac{b b^{\top}}{b^{\top}b} = \frac{1}{9} \left[ \begin{matrix} 1\\2\\2\end{matrix} \right] [1, 2, 2] = \frac{1}{9} \left[ \begin{matrix}1 & 2 &2 \\ 2 & 4 & 4 \\ 2 & 4 & 4\end{matrix}\right] . \tag{3.47}$$
> 现在我们选一个特定的向量 $x$，然后检查它的投影是否在这条直线上。不妨令 $x = [1, 1, 1]^{\top}$，然后计算它的投影：
> $$\pi_{U}(\boldsymbol{x}) = \boldsymbol{P}_{\pi}(x) = \frac{1}{9} \left[ \begin{matrix}1 & 2 &2 \\ 2 & 4 & 4 \\ 2 & 4 & 4\end{matrix} \right] \left[ \begin{matrix}1\\1\\1\end{matrix} \right] = \frac{1}{9 } \left[ \begin{matrix}5\\10\\10\end{matrix} \right] \in \text{span}\left\{ \left[ \begin{matrix}1\\2\\2\end{matrix} \right]  \right\} . \tag{3.48}$$
> 注意，$\boldsymbol{P}_{\pi}$ 作用在 $\pi_{U}(\boldsymbol{x})$ 上的结果等于它本身，这是说 $\boldsymbol{P}_{\pi}\pi_{U}(\boldsymbol{x}) = \pi_{U}(\boldsymbol{x})$。这并不令我们以外，因为根据定义 3.10，我们知道 $\boldsymbol{P}_{\pi}$ 是**幂等**的，也即对于任意的$x$，有 $\boldsymbol{P}_{\pi}^{2}x = \boldsymbol{P}_{\pi}$。

> Remark. With the results from Chapter 4, we can show that πU (x) is an eigenvector of P π, and the corresponding eigenvalue is 1.
> 注：在第四章，我们将证明 $\pi_{U}(\boldsymbol{x})$ 是矩阵 $\boldsymbol{P}_{\pi}$ 的一个特征向量，对应的特征值为 $1$。

### 3.8.2 向一般子空间投影

接下来我们讨论将向量 $x \in \mathbb{R}^{n}$ 投影至较低维度的一般子空间 $U \subset \mathbb{R}^{n}$，其中 $U$ 满足 $\dim U = m \geqslant 1$。如图 3.11 所示。

假设 $(b_{1}, \dots, b_{m})$ 是 $U$ 的一个正交基，$U$ 上的任意投影向量 $\pi_{U}(\boldsymbol{x})$ 必定是它的元素，因此 $U$ 中存在基向量 $b_{1}, \dots, b_{m}$ 的一个线性组合，满足 $\displaystyle \pi_{U}(\boldsymbol{x}) = \sum\limits_{i=1}^{m} \lambda_{i} b_{i}$。

> 注意，如果子空间 $U$ 是通过由一些向量张成的空间而给出的，读者在进行下面的计算之前需要确定其上的一个正交基 $b_{1}, \dots, b_{m}$。

和前文中投影至一维子空间类似，我们按照下面三步就可以找到投影向量 $\pi_{U}(\boldsymbol{x})$ 和投影矩阵 $\boldsymbol{P}_{\pi}$。

1. 确定投影向量在 $U$ 上的基下的坐标 $\lambda_{1}, \dots, \lambda_{m}$，使得下面的线性组合距离 $x \in \mathbb{R}^{n}$ 是最近的。$$\begin{align}\pi_{U}(\boldsymbol{x}) &= \sum\limits_{i=1}^{m} \lambda_{i}b_{i} = \boldsymbol{B\lambda}, \tag{3.49}\\\boldsymbol{B} &= [b_{1}, \dots, b_{m}] \in \mathbb{R}^{n \times m}, \boldsymbol{\lambda} = [\lambda_{1}, \dots, \lambda_{m}]^{\top} \in \mathbb{R}^{m} \tag{3.50}\end{align}$$
   和一维的例子一样，“最近”表示距离最短，这可以推断出连接 $\pi_{U}(\boldsymbol{x})$ 和 $x$ 的向量一定与 $U$ 的所有基向量都垂直（假设内积为点积）。$$\begin{align}\left\langle b_{1}, x - \pi_{U}(\boldsymbol{x}) \right\rangle =&~\, b_{1}^{\top}(x - \pi_{U}(\boldsymbol{x})) = 0, \tag{3.51}\\&\vdots\\\left\langle b_{m}, x - \pi_{U}(\boldsymbol{x}) \right\rangle =&~\, b_{m}^{\top}(x - \pi_{U}(\boldsymbol{x})) = 0, \tag{3.52}\\\end{align}$$
   依据 $\pi_{U}(\boldsymbol{x}) = \boldsymbol{B}\boldsymbol{\lambda}$ 对上式进行替换，有$$\begin{align}b_{1}^{\top}(x& - \boldsymbol{B\lambda}) = 0, \tag{3.53}\\&\vdots\\b_{m}^{\top}(x& - \boldsymbol{B\lambda}) = 0, \tag{3.53}\\\end{align}$$
   这样我们就得到了一个齐次线性方程$$\begin{align}\left[ \begin{matrix}b_{1}^{\top}\\ \vdots \\ b_{m}^{\top}\end{matrix} \right] (x - \boldsymbol{B\lambda}) & \iff \boldsymbol{B}^{\top}(x - \boldsymbol{B\lambda}) = 0 \tag{3.55}\\&\iff  \boldsymbol{B}^{\top}\boldsymbol{B\lambda} = \boldsymbol{B}^{\top}x \tag{3.56}\end{align}$$
   最后得到的方程 $(3.56)$ 叫**正规方程（normal equation）**。由于 $b_{1}, \dots, b_{m}$ 是 $U$ 的一个基，因此它们线性无关，所以矩阵 $\boldsymbol{B}^{\top}\boldsymbol{B} \in \mathbb{R}^{m \times m}$ 是正规矩阵，存在逆矩阵。所以我们可以求得解析解
    $$
    \boldsymbol{\lambda} = (\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}x. \tag{3.57}
    $$
    其中 $(\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}$ 叫矩阵 $\boldsymbol{B}$ 的**伪逆（pseudo-inverse）**，这对不是方形的矩阵也有效，唯一的要求就是 $\boldsymbol{B}^{\top}\boldsymbol{B}$ 是正定的，这表示 $\boldsymbol{B}$ 为**列满秩（full column rank）**。在实际操作中，我们常常对 $\boldsymbol{B}^{\top}\boldsymbol{B}$ 添加一个**摄动项（jitter term）** $\varepsilon \boldsymbol{I}, (\varepsilon > 0)$ 来满足其正定性和数值稳定性。这一对角线上的“山脊”将在第九章中使用Bayesian推断严格推导。
    
    > 译者注：揭示正规矩阵和摄动后的正规矩阵的正定性是显而易见的。任取 $x \in \mathbb{R}^{m}$，构造二次型 $x^{\top}B^{\top}Bx$，立刻有 $x^{\top}B^{\top}Bx = \|Bx\|_{2} \geqslant 0$，由范数的正定性知 $B^{\top}B$ 正定，因此满秩。
    > 对于摄动的情况，类似有 $x^{\top}(B^{\top}B + \varepsilon I)x = \|Bx\|_{2} + \varepsilon\|x\|_{2} > 0$，可知二次型严格大于零，因此摄动后的矩阵必然正定（满秩）。

2. 计算投影向量 $\pi_{U}(\boldsymbol{x}) \in U$。由于我们已经得到 $\pi_{U}(\boldsymbol{x}) = \boldsymbol{B}\boldsymbol{\lambda}$，因此由 $(3.57)$，有 $$\pi_{U}(\boldsymbol{x}) = \boldsymbol{B}(\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}x. \tag{3.58}$$
3. 计算投影矩阵 $\boldsymbol{P}_{\pi}$，从 $(3.58)$ 中我们可以立刻看出方程的解：
$$
\boldsymbol{P}_{\pi} = \boldsymbol{B} (\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}. \tag{3.59}
$$

> 注：上面对于至一般子空间的投影包含了一维的特殊情形。如果 $\dim U = 1$，则 $B^{\top}B \in \mathbb{R}$ 是一个标量，$(3.59)$ 可以被重写成 $\displaystyle \boldsymbol{P}_{\pi} = \frac{B B^{\top}}{B^{\top}B}$，这和 $(3.46)$ 中的矩阵完全一致。

> **示例 3.11（向二维子空间投影）**
> 对于子空间 $\displaystyle U = \text{span} \left\{ \begin{bmatrix} 1\\1\\1 \end{bmatrix}, \begin{bmatrix}0\\1\\2\end{bmatrix}\right\} \subset \mathbb{R}^{3}$ 和向量 $x = \begin{bmatrix}6\\0\\0\end{bmatrix} \in \mathbb{R}^{3}$，找到 $x$ 投影在 $U$ 中的坐标 $\boldsymbol{\lambda}$，投影向量 $\pi_{U}(\boldsymbol{x})$ 以及投影矩阵 $\boldsymbol{P}_{\pi}$
> ---
> 首先，我们检查张成 $U$ 的两个向量，发现它们线性无关，于是可以写成一个矩阵 $\boldsymbol{B} = \begin{bmatrix}1 & 0 \\ 1 & 1 \\ 1 & 2\end{bmatrix}$。
> 然后我们计算正规矩阵和 $\boldsymbol{x}$ 对两个向量的点积：$$\begin{align}\boldsymbol{B}^{\top}\boldsymbol{B} &= \left[ \begin{matrix} 1 & 1 & 1\\0 & 1 & 2\end{matrix} \right] \left[ \begin{matrix} 1 & 0\\1 & 1\\1 & 2\end{matrix} \right] = \left[ \begin{matrix} 3 & 3 \\ 3 & 5\end{matrix} \right], \\\boldsymbol{B}^{\top}x &= \left[ \begin{matrix} 1 & 1 & 1\\0 & 1 & 2\end{matrix} \right] \left[ \begin{matrix} 6\\0\\0\end{matrix} \right] = \left[ \begin{matrix} 6\\0\end{matrix} \right]. \end{align} \tag{3.60}$$
> 第三步，我们解正规方程 $\boldsymbol{B}^{\top}\boldsymbol{B\lambda} = \boldsymbol{B}^{\top}x$ 得到 $\boldsymbol{\lambda}$：$$\left[ \begin{matrix} 3 & 3 \\ 3 & 5\end{matrix} \right] \left[ \begin{matrix} \lambda_{1}\\\lambda_{2}\end{matrix} \right] = \left[ \begin{matrix} 6\\0\end{matrix} \right] \iff \boldsymbol{\lambda} = \left[ \begin{matrix}5\\-3\end{matrix} \right] $$
> 这样依赖，向量 $\boldsymbol{x}$ 投影至子空间 $U$ 的投影向量 $\pi_{U}(\boldsymbol{x})$，也就是向矩阵 $\boldsymbol{B}$ 的列空间投影的向量可以按下式直接进行计算：$$\pi_{U}(\boldsymbol{x}) = \boldsymbol{B\lambda} = \left[ \begin{matrix}5 \\ 2 \\ -1\end{matrix} \right]. \tag{3.62} $$
> 将原来的向量与投影后的向量作差得到向量的长度就是**投影损失（projection error）**：$$\|x - \pi_{U}(\boldsymbol{x})\| = \Big\|[1, -2, 1]^{\top}\Big\| = \sqrt{ 6 }. \tag{3.63}$$
> 相应地，对于任意 $\boldsymbol{x} \in \mathbb{R}^{3}$ 的投影矩阵 $\boldsymbol{P}_{\pi}$ 由下式给出：$$\boldsymbol{P}_{\pi} = \boldsymbol{B} (\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top} =- \frac{1}{6}\left[ \begin{matrix}5 & 2 & -1\\2 & 2 & 2\\-1&2&5\end{matrix} \right]. \tag{3.64}$$
> 我们可以通过验证残差向量 $x - \pi_{U}(\boldsymbol{x})$ 是否和所有 $U$ 的基垂直并考察 $\boldsymbol{P}_{\pi}^{2} = \boldsymbol{P}_{\pi}$ （参见定义 3.10）是否成立来验证计算结果的正确性。

> 注1：投影向量 $\pi_{U}(\boldsymbol{x})$ 虽然在子空间 $U \subset \mathbb{R}^{m}$ 中，但它依然是 $\mathbb{R}^{n}$ 中的向量。但我们只需用 $U$ 中关于基向量 $b_{1}, \dots, b_{m}$ 的坐标 $\lambda_{1}, \dots, \lambda_m$ 来表示它就足够了。

> 注2：在使用一般内积定义的向量空间中，我们在通过内积计算向量之间的夹角和距离是需要额外注意。

投影可以让我们对无解的线性系统 $\boldsymbol{Ax}  =\boldsymbol{b}$ 进行研究。让我们回忆 $\boldsymbol{b}$ 不在 $\boldsymbol{A}$ 张成的空间，也就是 $\boldsymbol{A}$ 所有列张成的空间（列空间）中的情形。在给出这样一个无解的线性系统时，我们可以找到一个**近似解**，也就是 $\boldsymbol{A}$ 的列空间中最接近 $\boldsymbol{b}$ 的向量。换句话说，我们计算 $\boldsymbol{b}$ 到 $\boldsymbol{A}$ 的列空间的投影，就是所求的近似解。这种问题在实作中非常常见，其得到的结果叫做超定系统（over-determined system）的**最小二乘估计（least-squares solution）**，类似地问题将在 9.4 节中继续讨论。如果再引入**重构损失（reconstruction error）**，就构成了推导主成分分析（10.3 节）的一种方式。

> 注：前文中我们只要求 $\{ \boldsymbol{b}_{1}, \dots, \boldsymbol{b}_{k} \}$ 是子空间 $U$ 的一个基，如果它是标准正交基，则 $(3.33)$ 和 $(3.34)$ 可以被用来化简 $(3.58)$。由于 $\boldsymbol{B}^{\top}\boldsymbol{B} = \boldsymbol{I}$，我们可以得到下面更加简洁的投影表达式：$$\pi_{U}(\boldsymbol{x}) = \boldsymbol{B B}^{\top}x \tag{3.65}$$
> 以及坐标 $\boldsymbol{\lambda}$ ：$$\boldsymbol{\lambda} = \boldsymbol{B}^{\top}x. \tag{3.66}$$
> 这意味着我们不再需要进行耗时的求逆计算了。

### 3.8.3 Gram-Schmidt 正交化

投影是 Gram-Schmidt正交化的核心，后者让我们可以从任意的 $n$ 维向量空间 $V$ 的一个基 $(\boldsymbol{b}_{1}, \dots, \boldsymbol{b}_{n})$ 构造出该空间的一个标准正交基 $(\boldsymbol{u}_{1}, \dots, \boldsymbol{u}_{n})$ 。这个正交基总是存在，且满足 $\text{span}\{ \boldsymbol{b}_{1}, \dots, \boldsymbol{b}_{n} \} = \text{span}\{ \boldsymbol{u}_{1}, \dots, \boldsymbol{u}_{n} \}$。所谓的 Gram-Schmidt 正交化方法在给定 $V$ 的任意基 $(\boldsymbol{b}_{1}, \dots, \boldsymbol{b}_{n})$ 的情况下迭代地构造出正交基 $(\boldsymbol{u}_{1}, \dots, \boldsymbol{u}_{n})$，其过程如下：
$$
\begin{align}
\boldsymbol{u}_{1} &:= \boldsymbol{b}_{1}, \tag{3.67}\\
\boldsymbol{u}_{k} &:= \boldsymbol{b}_{k} - \pi_{\text{span}\{ \boldsymbol{u}_{1}, \dots, \boldsymbol{u}_{k-1} \}}(\boldsymbol{b}_{k}), \quad k = 2, \dots, n.
\end{align}
$$
在式 $(3.68)$ 中，第 $k$ 个基向量 $\boldsymbol{b}_{k}$ 被投影至前 $k-1$ 个构造得到的单位正交向量 $\boldsymbol{u}_{1}, \dots, \boldsymbol{u}_{k-1}$ 张成的子空间上（参见 3.8.2 节）。向量 $\boldsymbol{b}_{k}$ 减去这个投影向量所得的向量 $\boldsymbol{u}_{k}$ 与 $\boldsymbol{u}_{1}, \dots, \boldsymbol{u}_{k-1}$ 张成的 $k-1$ 维子空间垂直。对所有 $n$ 个基向量 $\boldsymbol{b}_{1}, \dots, \boldsymbol{b}_{n}$ 逐个应用这个算法，就得到了空间 $V$ 的一个正交基 $(\boldsymbol{u}_{1}, \dots, \boldsymbol{u}_{n})$ 。如果我们将正交基中的向量全部标准化，使得对所有的 $k = 1, \dots, n$ 都有 $\|\boldsymbol{u}_{k}\| = 1$，我们就得到了原空间的一个标准正交基（ONB）。

> **示例 3.12（Gram-Schmidt 正交化）**
> ![](Pasted%20image%2020240817143850.png)
> <center>图 3.12 Gram-Schmidt 正交化</center>
> 如图 3.12 所示，考虑 $\mathbb{R}^{2}$ 的一个基 $(\boldsymbol{b}_{1}, \boldsymbol{b}_{2})$，其中
> $$\boldsymbol{b}_{1} = \begin{bmatrix}2\\0\end{bmatrix}, \quad \boldsymbol{b}_{2} = \begin{bmatrix}1\\1\end{bmatrix}; \tag{3.69}$$
> 使用 Gram-Schmidt 正交化方法，我们可按照下面的过程构造 $\mathbb{R}^{2}$ 的一个正交基：$$\begin{align}\boldsymbol{u}_{1} &= \boldsymbol{b}_{1} = \begin{bmatrix}2\\0\end{bmatrix},\tag{3.70}\\\boldsymbol{u}_{2} &= \boldsymbol{b_{2}} - \pi_{\text{span}\{ \boldsymbol{u}_{1} \}}(\boldsymbol{b}_{2}) \\ &\,\mathop{=\!=\!=}\limits^{(3.45)} \,\boldsymbol{b}_{2} - \frac{\boldsymbol{u}_{1}\boldsymbol{u}_{1}^{\top}}{\|\boldsymbol{u}_{1}\|^{2}}\cdot\boldsymbol{b}_{2} = \begin{bmatrix}1\\1\end{bmatrix} - \begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}0\\1\end{bmatrix}. \tag{3.71}\end{align}$$
> 上面的步骤对应图 3.12 中的 (b) 和 (c)。我们可以立即看出 $\boldsymbol{u}_{1}$ 和 $\boldsymbol{u}_{2}$ 是垂直的，也即 $\boldsymbol{u}_{1} ^\top \boldsymbol{u}_{2} = 0$。

### 3.8.4 向仿射子空间投影

直到现在我们讨论的都是如何讲一个向量投影到低维的子空间 $U$ 上。本节将讨论如何解决投影至仿射子空间的问题。

![](Pasted%20image%2020240817145017.png)
<center>图 3.13 向仿射空间投影</center>

考虑像图 3.13(a) 这样的问题：给定一个仿射空间 $L = \boldsymbol{x}_{0} + U$，其中 $\boldsymbol{b}_{1}, \boldsymbol{b_{2}}$ 是 $U$ 的一个基。为确定向量 $\boldsymbol{x}$ 到仿射空间 $L$ 的投影 $\pi_{L}(\boldsymbol{x})$，我们选择将其转换为我们已解决的投影至低维子空间的问题。我们对 $\boldsymbol{x}$ 和 $L$ 同时减去支持点 $\boldsymbol{x}_{0}$，这样一来 $L - \boldsymbol{x}_{0}$ 恰好就是子空间 $U$。这样我们可以使用前文中 3.8.2 节讨论的正交投影至子空间的方法得到 $\pi_{U}(\boldsymbol{x} - \boldsymbol{x}_{0})$（如图 3.13 (b) 所示），然后我们可以把 $\boldsymbol{x}_{0}$ 加回投影向量，将它重新放入 $L$ 中，这样我们就得到了 $\boldsymbol{x}$ 到 $L$ 的投影：
$$
\pi_{L}(\boldsymbol{x}) = \boldsymbol{x}_{0} + \pi_{U}(\boldsymbol{x} - \boldsymbol{x}_{0}), \tag{3.72}
$$
其中 $\pi_{U}(\cdot)$ 是至子空间 $U$ 的投影，也就是 $L$ 的方向空间（如图3.13所示）。

从图可以看出，从 $\boldsymbol{x}$ 到 $L$ 的距离和 $\boldsymbol{x} - \boldsymbol{x}_{0}$ 到 $U$ 的距离相等，也就是
$$
\begin{align}
d(\boldsymbol{x}, L) &= \|\boldsymbol{x} - \pi_{L}(\boldsymbol{x})\| = \|\boldsymbol{x} - [\boldsymbol{x}_{0} + \pi_{U}(\boldsymbol{x} - \boldsymbol{x}_{0})]\| \tag{3.73a}\\[0.2em]
&= d(\boldsymbol{x} - \boldsymbol{x}_{0}, \pi_{U}(\boldsymbol{x} - \boldsymbol{x}_{0})) = d(\boldsymbol{x} - \boldsymbol{x}_{0}, U). \tag{3.73b}
\end{align}
$$
在 12.1 节，我们将会用这个方法导出**分割超平面**这个概念。

## 3.9 旋转

### 3.9.1 $\mathbb{R}^{2}$中的旋转

### 3.9.2 $\mathbb{R}^{3}$中的旋转

### 3.9.3 $n$维空间中的旋转

### 3.9.4 旋转算子的性质

## 3.10 拓展阅读
