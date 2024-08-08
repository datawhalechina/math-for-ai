
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
> $$ \begin{align} \| \cdot \|: V &\rightarrow \mathbb{R} \tag{3.1}\\ x &\mapsto \| x \|,  \tag{3.2}\end{align} $$
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
> $$\forall x \in V \textbackslash \{ 0 \}: \Omega(x, x) > 0, ~~ ~~ \Omega(0, 0) = 0,\tag{3.8}$$
>   则称$\Omega$为*正定*的。

![500](Pasted%20image%2020240302120513.png)
> **定义 3.3**
> 设$V$为向量空间，双线性映射$\Omega:  V \times V \rightarrow \mathbb{R}$将两个$V$中的向量映射到一个实数，则
> * 对称且正定的双线性映射$\Omega$被称为$V$上的一个*内积*，并简写$\Omega(x, y)$为$\left\langle x, y \right\rangle$。
> * 二元组$(V, \left\langle \cdot, \cdot \right\rangle)$称为*内积空间*或*装配有内积的（实）向量空间*。特别地，如果内积采用（式 3.5）中定义的点积，则称$(V, \left\langle \cdot, \cdot \right\rangle)$为欧几里得向量空间（译者注：简称欧氏空间）

We will refer to these spaces as inner product spaces in this book.
本书中我们称这些空间为内积空间。

![500](Pasted%20image%2020240302120624.png)
> **示例 3.3**（不是点积的内积）
> 考虑$V = \mathbb{R}^{2}$，定义下面的内积：
> $$\left\langle x, y \right\rangle := x_{1}y_{1} - (x_{1}y_{2} + x_{2}y_{1}) + 2x_{2}y_{2},\tag{3.9}$$
> 可以验证这是一个与点积不同的内积，证明留作练习。

### 3.2.3 对称和正定矩阵
Symmetric, positive definite matrices play an important role in machine learning, and they are defined via the inner product. 
对称和正定矩阵在机器学习中十分重要，它们是由内积定义的。

In Section 4.3, we will return to symmetric, positive definite matrices in the context of matrix decompositions. 
在 4.3 节中，我们在讨论矩阵分解时将会回到这个概念。

The idea of symmetric positive semidefinite matrices is key in the definition of kernels (Section 12.4).
在 12.4 节中，对称和半正定矩阵还在核的定义中起到关键作用。

Consider an n-dimensional vector space V with an inner product h·, ·i :× V → R (see Definition 3.3) and an ordered basis B = (b1, . . . , bn) of V . 
假设$n$维线性空间$V$装配有内积$\left\langle \cdot, \cdot \right\rangle: V \times V \rightarrow \mathbb{R}$（参见定义 3.3）并取$V$中的一个基（已排序）$B = (b_{1}, \dots, b_{n})$，

Recall from Section 2.6.1 that any vectors x, y ∈ V can be written as linear combinations of the basis vectors so that x = P n i=1 ψibi ∈ V and y = P n j=1 λjbj ∈ V for suitable ψi , λj ∈ R. 
在 2.6.1 节中我们知道任意$x, y \in V$，可以找到$\lambda_{i}, \psi_{i} \in \mathbb{R}, i=1,\dots,n$，使得两个向量可以写成基$B$中向量的线性组合，即$\displaystyle x = \sum\limits_{i=1}^{n} \psi_{i}b_{i} \in V$，$\displaystyle y = \sum\limits_{j=1}^{n} \lambda_{j}b_{j} \in V$。

Due to the bilinearity of the inner product, it holds for all x, y ∈ V that
由内积的双线性性，对所有的$x, y \in V$，有
$$
\left\langle x, y \right\rangle =
\left\langle \sum\limits_{i=1}^{n} \psi_{i}b_{i}, \sum\limits_{j=1}^{n} \lambda_{j}b_{j} \right\rangle 
= \sum\limits_{i=1}^{n} \sum\limits_{j=1}^{n} \psi_{i} \lambda_{j} \left\langle b_{i}, b_{j} \right\rangle = \hat{x}^{\top} A \hat{y},
$$

where Aij := h bi, bj i and ˆx, ˆy are the coordinates of x and y with respect to the basis B
其中$A_{i,j} := \left\langle b_{i}, b_{j} \right\rangle$（译者注：这就是线性空间$V$中的一个*度量矩阵*），$\hat{x}$和$\hat{y}$为原向量在基$B$下的*坐标*。

 This implies that the inner product h·, ·i is uniquely determined through A.
这意味着内积$\left\langle \cdot, \cdot \right\rangle$被矩阵$A$*唯一确定*，

The symmetry of the inner product also means that A is symmetric.
且由于内积具有对称性，不难看出$A$是对称矩阵。

Furthermore, the positive definiteness of the inner product implies that
进一步地，根据内积的正定性，我们可以得出下面的结论：
$$
\forall x \in V \textbackslash \{ 0 \}: x^{\top}Ax > 0. \tag{3.11}
$$
![500](Pasted%20image%2020240302125526.png)
> **定义 3.4**（对称正定矩阵）
> 一个$n$级对称矩阵$A \in \mathbb{R}^{n \times n}$若满足（式 3.11），则被称为*对称正定矩阵*（或仅称为正定矩阵）。如果只满足将（式 3.11）中的不等号改成$\geqslant$的条件，则称为*对称半正定矩阵*

![500](Pasted%20image%2020240302125539.png)
> **示例 3.4**（对称正定矩阵）
> 考虑下面两个矩阵
> $$A_{1} = \left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right] , \quad A_{2} = \left[  \begin{matrix} 9 & 6 \\ 6 & 3 \end{matrix} \right],\tag{3.12}$$
> 其中$A_{1}$是对称且正定的，因为它不仅对称（译者注：这显而易见），而且对于任意$x \in \mathbb{R}^{2} \textbackslash \{ 0 \}$都有，
> $$\begin{align} x^{\top}A_{1}x &= \left[ \begin{matrix} x_{1} & x_{2} \end{matrix}\right]\left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right]\left[ \begin{matrix} x_{1} \\ x_{2}  \end{matrix}\right] \\\ &= 9x_{1}^{2} + 12x_{1}x_{2} + 5x_{2}^{2} \\ &= (3x_{1} + 2x_{2})^{2} + x_{2}^{2} > 0.\end{align} \tag{3.13}$$
> 相反地，$A_{2}$不是正定矩阵。如果取$x = [2, -3]^{\top}$，可以验证二次型$x^\top Ax$是负数。


If A ∈ Rn×n is symmetric, positive definite, then \[formula 3.15\] defines an inner product with respect to an ordered basis B, where ˆx and
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
如图3.4所示，在$[0,\pi]$中有唯一的$\omega$满足下面的等式：
$$
\cos\omega = \frac{\left\langle x, y \right\rangle}{\|x\| \|y\|}. \tag{3.25}
$$

![300](Pasted%20image%2020240627164202.png)
Figure 3.4 When restricted to [0, π] then f(ω) = cos(ω) returns a unique number in the interval [−1, 1].
图3.4 定义域为$[0,\pi]$时的余弦函数图像，此时角度值和余弦函数值一一对应

The number ω is the angle between the vectors x and y. Intuitively, the angle angle between two vectors tells us how similar their orientations are. For example, using the dot product, the angle between x and y = 4x, i.e., y is a scaled version of x, is 0: Their orientation is the same.
而$\omega$就是$x$和$y$之间的夹角。直观意义上，两向量之间的夹角给出了其方向的相似程度，例如两向量$x$和$y=4x$（$x$经过常数缩放后的版本）的夹角为零，因此它们的方向相同。

> Example 3.6 (Angle between Vectors)
> Let us compute the angle between x = [1, 1]> ∈ R2 and y = [1, 2]> ∈ R2 ; see Figure 3.5, where we use the dot product as the inner product. Then we get
> **示例 3.6 （向量之间的夹角）**
> 如图3.5所示，计算向量$x = [1, 1]^{\top} \in \mathbb{R}^{2}$和$y = [1, 2]^{\top} \in \mathbb{R}^{2}$的夹角。我们令向量的内积为点积，有
> $$ \cos\omega = \frac{\left\langle x, y \right\rangle}{\sqrt{ \left\langle x, x \right\rangle \left\langle y, y \right\rangle  }} = \frac{x^{\top}y}{\sqrt{ x^{\top}xy^{\top}y }} = \frac{3}{\sqrt{ 10 }}, \tag{3.26} $$
> and the angle between the two vectors is arccos( √ 3  10 ) ≈ 0.32 rad, which corresponds to about 18◦
> 于是两个向量的夹角余弦值为$\displaystyle \arccos\left( \frac{3}{\sqrt{ 10 }} \right) \approx 0.32\text{ rad}$，大约为$18^{\circ}$。
> ![150](Pasted%20image%2020240627165050.png)
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
> ![300](Pasted%20image%2020240627170238.png)
> 图3.6 使用不同的内积定义计算得到的两向量$x$和$y$之间的夹角不同
> 如图3.6所示，考虑向量$x=[1,1]^{\top}, y = [-1, 1]^{\top} \in \mathbb{R}^{2}$，考虑它们在不同内积定义下的夹角大小。如果使用通常的点积作为内积，则它们之间的夹角为$90^{\circ}$，也即$x \bot y$。但如果使用下面的内积定义则会得到不同的结果： $$ \left\langle x, y \right\rangle  = x^{\top} \left[ \begin{matrix} 2 & 0 \\ 0 & 1
\end{matrix}\right] y, \tag{3.27} $$
> we get that the angle ω between x and y is given by
> 可以计算得到在这个内积之下两向量之间的夹角为> $$ \cos \omega = \frac{\left\langle x, y \right\rangle}{\|x\| \|y\|} = -\frac{1}{3} \implies \omega \approx 1.91 \text{ rad} \approx 109.5^{\circ}, \tag{3.28} $$
> and x and y are not orthogonal. Therefore, vectors that are orthogonal with respect to one inner product do not have to be orthogonal with respect to a different inner product.
> 于是$x$和$y$并不正交。因此在一个内积下正交的两个向量在另一个内积下不一定正交。


> Definition 3.8 (Orthogonal Matrix). 
> A square matrix A ∈ Rn×n is an orthogonal matrix if and only if its columns are orthonormal so that
>  **定义3.8（正交矩阵）**
>  方阵$A \in \mathbb{R}^{n \times n}$为正交矩阵当且仅当满足下面的条件：
>  $$ A A^{\top} = I = A^{\top} A, \tag{3.29} $$
>  which implies that
>  进而有
>  $$ A^{-1} = A^{\top}, \tag{3.30} $$
> i.e., the inverse is obtained by simply transposing the matrix. 
> 这是说，正交矩阵的逆是它的转置。

Remark：It is convention to call these matrices “orthogonal” but a more precise description would be “orthonormal”. Transformations with orthogonal matrices preserve distances and angles.
注：一般我们将这些矩阵称为“orthogonal matrix”，严格意义上它们应该被称为“orthonormal matrix”。因为“orthonormal matrix”对应的变换在线性空间内保持向量的长度和向量之间的夹角。译者注：中文中不做区分，统一称为“正交矩阵”。正交矩阵对应的变换在$\mathbb{R}^{2}$和$\mathbb{R}^{3}$中属于刚体变换。

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

$$
\begin{align}
\left\langle b_{i}, b_{j} \right\rangle &= 0, & i \ne j \tag{3.33}\\
\left\langle b_{i}, b_{i} \right\rangle &= 1 \tag{3.34}
\end{align}
$$

$$
b_{1} = \frac{1}{\sqrt{ 2 }} \left[ \begin{matrix}
1\\1
\end{matrix} \right], \quad
b_{2} = \frac{1}{\sqrt{ 2 }} \left[ \begin{matrix}
1\\-1
\end{matrix} \right] \tag{3.35}
$$

## 3.6 正交补


$$
x = \sum\limits_{m=1}^{M} \lambda b_{m} + \sum\limits_{j=1}^{D-M} \psi_{i}b_{j}^{\bot}, \quad \lambda_{m}, \psi_{j} \in \mathbb{R} \tag{3.36}
$$

## 3.7 函数的点积

$$
\left\langle u, v \right\rangle := \int_{a}^{b} {u(x)v(x)} \, \mathrm d{x} \tag{3.37} 
$$


$$
\{ 1, \cos(x), \cos(2x), \cos(3x), \dots \} \tag{3.38}
$$



## 3.8 正交投影
### 3.8.1 向一维子空间（直线）投影

$$
\left\langle x - \pi_{U}(x), b \right\rangle = 0 \mathop{\iff}\limits^{\pi_{U}(x) = \lambda b} \left\langle x - \lambda b, b \right\rangle = 0. \tag{3.39}  
$$

$$
\left\langle x, b \right\rangle - \lambda\left\langle b, b \right\rangle = 0 \iff \lambda = \frac{\left\langle x, b \right\rangle}{\left\langle b, b \right\rangle } = \frac{\left\langle x, b \right\rangle}{\|b\|^{2}} = 0 . \tag{3.40}
$$
$$
\lambda = \frac{b^{\top}x}{b^{\top}b} = \frac{b^{\top}x}{\|b\|^{2}}. \tag{3.41}
$$

$$
\pi_{U}(x) = \lambda b = \frac{\left\langle x, b \right\rangle}{\|b\|^{2}} \cdot b = \frac{b^{\top}x}{\|b\|^{2}} \cdot b, \tag{3.42}
$$

$$
\|\pi_{U}(x)\| = \|\lambda b\| = |\lambda| \|b\|. \tag{3.43}
$$

$$
\|\pi_{U}(x)\| ~\mathop{=\!=\!=}\limits^{(3.42)} ~\frac{|b^{\top}x|}{\|b\|^{2}} \|b\|~ \mathop{=\!=\!=}\limits^{(3.25)} ~|\!\cos\omega| \cdot \|x\| \cdot \|b\| \cdot \frac{\|b\|}{\|b\|^{2}} = |\!\cos{\omega}| \cdot \|x\|. \tag{3.44}
$$

$$
\pi_{U}(x) = \lambda b = b\lambda =b \frac{b^{\top}x}{\|b\|^{2}} = \frac{bb^{\top}}{\|b\|^{2}} x, \tag{3.45}
$$

$$
P_{\pi} = \frac{b b^{\top}}{\|b\|^{2}}. \tag{3.46}
$$

$$
P_{\pi} = \frac{b b^{\top}}{b^{\top}b} = \frac{1}{9} \left[ \begin{matrix} 1\\2\\2
\end{matrix} \right] [1, 2, 2] = \frac{1}{9} \left[ \begin{matrix}
1 & 2 &2 \\ 2 & 4 & 4 \\ 2 & 4 & 4
\end{matrix} \right] . \tag{3.47}
$$

$$
\pi_{U}(x) = P_{\pi}(x) = \frac{1}{9} \left[ \begin{matrix}
1 & 2 &2 \\ 2 & 4 & 4 \\ 2 & 4 & 4
\end{matrix} \right] \left[ \begin{matrix}
1\\1\\1
\end{matrix} \right] = \frac{1}{9 } \left[ \begin{matrix}
5\\10\\10
\end{matrix} \right] \in \text{span}\left\{ \left[ \begin{matrix}
1\\2\\2
\end{matrix} \right]  \right\} . \tag{3.48}
$$

### 3.8.2 向一般子空间投影

$$
\begin{align}
\pi_{U}(x) &= \sum\limits_{i=1}^{m} \lambda_{i}b_{i} = \boldsymbol{B\lambda}, \tag{3.49}\\
\boldsymbol{B} &= [b_{1}, \dots, b_{m}] \in \mathbb{R}^{n \times m}, \boldsymbol{\lambda} = [\lambda_{1}, \dots, \lambda_{m}]^{\top} \in \mathbb{R}^{m} \tag{3.50}
\end{align}
$$

$$
\begin{align}
\left\langle b_{1}, x - \pi_{U}(x) \right\rangle =&~\, b_{1}^{\top}(x - \pi_{U}(x)) = 0, \tag{3.51}\\
&\vdots\\
\left\langle b_{m}, x - \pi_{U}(x) \right\rangle =&~\, b_{m}^{\top}(x - \pi_{U}(x)) = 0, \tag{3.52}\\
\end{align}
$$

$$
\begin{align}
b_{1}^{\top}(x& - \boldsymbol{B\lambda}) = 0, \tag{3.53}\\
&\vdots\\
b_{m}^{\top}(x& - \boldsymbol{B\lambda}) = 0, \tag{3.53}\\
\end{align}
$$

$$
\begin{align}
\left[ \begin{matrix}
b_{1}^{\top}\\ \vdots \\ b_{m}^{\top}
\end{matrix} \right] (x - \boldsymbol{B\lambda}) & \iff B^{\top}(x - B\lambda) = 0 \tag{3.55}\\
&\iff  \boldsymbol{B}^{\top}\boldsymbol{B\lambda} = \boldsymbol{B}^{\top}x \tag{3.56}
\end{align}
$$

$$
\boldsymbol{\lambda} = (\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}x. \tag{3.57}
$$

$$
\pi_{U}(x) = \boldsymbol{B}(\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}x. \tag{3.58}
$$

$$
P_{\pi} = \boldsymbol{B} (\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}
$$

$$
\boldsymbol{B}^{\top}\boldsymbol{B} = \left[ \begin{matrix} 1 & 1 & 1\\0 & 1 & 2
\end{matrix} \right] \left[ \begin{matrix} 1 & 0\\1 & 1\\1 & 2
\end{matrix} \right] = \left[ \begin{matrix} 3 & 3 \\ 3 & 5
\end{matrix} \right],\quad \boldsymbol{B}^{\top}x = \left[ \begin{matrix} 1 & 1 & 1\\0 & 1 & 2
\end{matrix} \right] \left[ \begin{matrix} 6\\0\\0
\end{matrix} \right] = \left[ \begin{matrix} 6\\0
\end{matrix} \right]. \tag{3.60}
$$

$$
\left[ \begin{matrix} 3 & 3 \\ 3 & 5
\end{matrix} \right] \left[ \begin{matrix} \lambda_{1}\\\lambda_{2}
\end{matrix} \right] = \left[ \begin{matrix} 6\\0
\end{matrix} \right] \iff \boldsymbol{\lambda} = \left[ \begin{matrix}
5\\-3
\end{matrix} \right] 
$$

$$
\pi_{U}(x) = \boldsymbol{B\lambda} = \left[ \begin{matrix}
5 \\ 2 \\ -1
\end{matrix} \right]. \tag{3.62} 
$$

$$
\|x - \pi_{U}(x)\| = \Big\|[1, -2, 1]^{\top}\Big\| = \sqrt{ 6 }. \tag{3.63}
$$

$$
P_{\pi} = \boldsymbol{B} (\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top} =- \frac{1}{6}\left[ \begin{matrix}
5 & 2 & -1\\2 & 2 & 2\\-1&2&5
\end{matrix} \right]. \tag{3.64}
$$

$$
\pi_{U}(x) = \boldsymbol{B B}^{\top}x \tag{3.65}
$$

$$
\boldsymbol{\lambda} = \boldsymbol{B}^{\top}x. \tag{3.66}
$$


### 3.8.3 Gram-Schmidt 正交化


### 3.8.4 向仿射子空间投影


## 3.9 旋转
### 3.9.1 $\mathbb{R}^{2}$中的旋转


### 3.9.2 $\mathbb{R}^{3}$中的旋转


### 3.9.3 $n$维空间中的旋转


### 3.9.4 旋转算子的性质


## 3.10 拓展阅读

