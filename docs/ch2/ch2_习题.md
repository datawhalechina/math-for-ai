

## 练习 2.1
我们考虑 $(\mathbb{R}\backslash\{-1\}, \star)$，其中

$$ 
a \star b := ab + a + b, \quad a, b \in \mathbb{R}\backslash\{-1\} 
$$

a. 证明 $(\mathbb{R}\backslash\{-1\}, \star)$ 是一个阿贝尔群。
b. 在阿贝尔群 $(\mathbb{R}\backslash\{-1\}, \star)$ 中解方程

$$ 
3 \star x \star x = 15 
$$

## 练习 2.2
设 $n$ 是 $\mathbb{N}\backslash\{0\}$ 中的元素。设 $k, x$ 是 $\mathbb{Z}$ 中的元素。我们定义整数 $k$ 的同余类 $\bar{k}$ 为集合

$$ 
\bar{k} = \{x \in \mathbb{Z} \mid x - k = 0 \pmod{n}\} 
$$

$$ 
= \{x \in \mathbb{Z} \mid \exists a \in \mathbb{Z}, \text{ 使得 } x - k = n \cdot a\} 
$$

我们现在定义 $\mathbb{Z}/n\mathbb{Z}$（有时写作 $\mathbb{Z}_n$）为所有模 $n$ 的同余类的集合。欧几里得除法表明这个集合是一个包含 $n$ 个元素的有限集：

$$ 
\mathbb{Z}_n = \{0, 1, \ldots, n - 1\} 
$$

对于所有 $a, b \in \mathbb{Z}_n$，我们定义

$$ 
a \oplus b := a + b 
$$

a. 证明 $(\mathbb{Z}_n, \oplus)$ 是一个群。它是阿贝尔群吗？
b. 现在我们为所有 $a$ 和 $b$ 在 $\mathbb{Z}_n$ 中定义另一个运算 $\otimes$：

$$ 
a \otimes b = a \times b 
$$

其中 $a \times b$ 表示 $\mathbb{Z}$ 中的通常乘法。设 $n = 5$。绘制 $\mathbb{Z}_5\backslash\{0\}$ 中元素在 $\otimes$ 下的乘法表，即计算所有 $a$ 和 $b$ 在 $\mathbb{Z}_5\backslash\{0\}$ 中的乘积 $a \otimes b$。由此，证明 $\mathbb{Z}_5\backslash\{0\}$ 在 $\otimes$ 下是封闭的，并且存在单位元。列出 $\mathbb{Z}_5\backslash\{0\}$ 中所有元素在 $\otimes$ 下的逆元。得出结论：$(\mathbb{Z}_5\backslash\{0\}, \otimes)$ 是一个阿贝尔群。
c. 证明 $(\mathbb{Z}_8\backslash\{0\}, \otimes)$ 不是一个群。
d. 回忆贝祖定理指出，两个整数 $a$ 和 $b$ 互质（即 $\gcd(a, b) = 1$）当且仅当存在两个整数 $u$ 和 $v$ 使得 $au + bv = 1$。证明 $(\mathbb{Z}_n\backslash\{0\}, \otimes)$ 是一个群当且仅当 $n \in \mathbb{N}\backslash\{0\}$ 是质数。

## 练习 2.3
考虑以下定义的 $3 \times 3$ 矩阵集合 $G$：

$$ 
G = \left\{ \begin{pmatrix} 1 & x & z \\ 0 & 1 & y \\ 0 & 0 & 1 \end{pmatrix} \in \mathbb{R}^{3 \times 3} \,\Bigg|\, x, y, z \in \mathbb{R} \right\} 
$$

我们将 $\cdot$ 定义为标准矩阵乘法。$(G, \cdot)$ 是一个群吗？如果是，它是阿贝尔群吗？请证明你的答案。

## 练习 2.4
计算以下矩阵乘积（如果可能的话）：

a.

$$ 
\begin{pmatrix} 1 & 2 & 4 \\ 5 & 7 & 8 \end{pmatrix} \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{pmatrix} 
$$

b.

$$ 
\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix} \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{pmatrix} 
$$

c.

$$ 
\begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix} 
$$

d.

$$ 
\begin{pmatrix} 1 & 2 \\ 1 & 2 \\ 4 & 1 \\ -1 & -4 \end{pmatrix} \begin{pmatrix} 0 & 3 & 1 & -1 \\ 2 & 1 & 5 & 2 \end{pmatrix}
$$

e.

$$ 
\begin{pmatrix} 0 & 3 & 1 & -1 \\ 2 & 1 & 5 & 2 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 1 & 2 \\ 4 & 1 \\ -1 & -4 \end{pmatrix} 
$$

## 练习 2.5
求解以下非齐次线性方程组 $Ax = b$ 的所有解，其中 $A$ 和 $b$ 定义如下：

a.

$$ 
A = \begin{pmatrix} 1 & 1 & -1 & -1 \\ 2 & 5 & -7 & -5 \\ 2 & -1 & 1 & 3 \\ 5 & 2 & -4 & 2 \end{pmatrix}, \quad b = \begin{pmatrix} 1 \\ -2 \\ 4 \\ 6 \end{pmatrix} 
$$

b.

$$ 
A = \begin{pmatrix} 1 & -1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ -3 & 0 & 2 & -1 \\ 0 & 1 & -1 & -1 \\ 2 & 0 & -2 & -1 \end{pmatrix}, \quad b = \begin{pmatrix} 3 \\ 6 \\ 5 \\ -1 \end{pmatrix} 
$$

## 练习 2.6
使用高斯消元法，求解非齐次方程组 $Ax = b$ 的所有解，其中

$$ 
A = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}, \quad b = \begin{pmatrix} 2 \\ -1 \\ 1 \end{pmatrix} 
$$

## 练习 2.7
求解方程组 $Ax = 12x$ 的所有解 $x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} \in \mathbb{R}^3$，其中

$$ 
A = \begin{pmatrix} 6 & 4 & 3 \\ 6 & 0 & 9 \\ 0 & 8 & 0 \end{pmatrix} 
$$

并且满足 $\sum_{i=1}^3 x_i = 1$。

## 练习 2.8
如果可能的话，求以下矩阵的逆矩阵：

a.

$$ 
A = \begin{pmatrix} 2 & 3 & 4 \\ 3 & 4 & 5 \\ 4 & 5 & 6 \end{pmatrix} 
$$

b.

$$ 
A = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{pmatrix} 
$$

## 练习 2.9
以下哪些集合是 $\mathbb{R}^3$ 的子空间？
a. $A = \{(\lambda, \lambda + \mu^3, \lambda - \mu^3) \mid \lambda, \mu \in \mathbb{R}\}$
b. $B = \{(\lambda^2, -\lambda^2, 0) \mid \lambda \in \mathbb{R}\}$
c. 设 $\gamma \in \mathbb{R}$，$C = \{(\xi_1, \xi_2, \xi_3) \in \mathbb{R}^3 \mid \xi_1 - 2\xi_2 + 3\xi_3 = \gamma\}$
d. $D = \{(\xi_1, \xi_2, \xi_3) \in \mathbb{R}^3 \mid \xi_2 \in \mathbb{Z}\}$

## 练习 2.10
以下向量集合是否线性无关？

a.

$$ 
x_1 = \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}, \quad x_2 = \begin{pmatrix} 1 \\ 1 \\ -2 \end{pmatrix}, \quad x_3 = \begin{pmatrix} 3 \\ -3 \\ 8 \end{pmatrix} 
$$

b.

$$ 
x_1 = \begin{pmatrix} 1 \\ 2 \\ 1 \\ 0 \\ 0 \end{pmatrix}, \quad x_2 = \begin{pmatrix} 1 \\ 1 \\ 0 \\ 1 \\ 1 \end{pmatrix}, \quad x_3 = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 1 \\ 1 \end{pmatrix} 
$$

## 练习 2.11
将

$$ 
y = \begin{pmatrix} 1 \\ -2 \\ 5 \end{pmatrix} 
$$

表示为以下向量的线性组合：

$$ 
x_1 = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}, \quad x_2 = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}, \quad x_3 = \begin{pmatrix} 2 \\ -1 \\ 1 \end{pmatrix} 
$$

## 练习 2.12
考虑 $\mathbb{R}^4$ 的两个子空间：

$$ 
U_1 = \text{span}\left( \begin{pmatrix} 1 \\ 1 \\ -3 \\ 1 \end{pmatrix}, \begin{pmatrix} 2 \\ -1 \\ 0 \\ -1 \end{pmatrix}, \begin{pmatrix} -1 \\ 1 \\ -1 \\ 1 \end{pmatrix} \right) 
$$

$$ 
U_2 = \text{span}\left( \begin{pmatrix} -1 \\ -2 \\ 2 \\ 1 \end{pmatrix}, \begin{pmatrix} 2 \\ -2 \\ 0 \\ 0 \end{pmatrix}, \begin{pmatrix} -3 \\ 6 \\ -2 \\ -1 \end{pmatrix} \right) 
$$

确定 $U_1 \cap U_2$ 的一个基。

## 练习 2.13
考虑两个子空间 $U_1$ 和 $U_2$，其中 $U_1$ 是齐次方程组 $A_1x = 0$ 的解空间，$U_2$ 是齐次方程组 $A_2x = 0$ 的解空间，其中

$$ 
A_1 = \begin{pmatrix} 1 & 0 & 1 & 1 \\ -2 & -1 & 2 & 1 \\ 3 & 1 & 0 & 1 \end{pmatrix}, \quad A_2 = \begin{pmatrix} 3 & -3 & 0 & 1 \\ 2 & 3 & 7 & -5 \\ 2 & 3 & -1 & 2 \end{pmatrix} 
$$

a. 确定 $U_1$ 和 $U_2$ 的维数。
b. 确定 $U_1$ 和 $U_2$ 的基。
c. 确定 $U_1 \cap U_2$ 的一个基。

## 练习 2.14
考虑两个子空间 $U_1$ 和 $U_2$，其中 $U_1$ 由 $A_1$ 的列向量生成，$U_2$ 由 $A_2$ 的列向量生成，其中

$$ 
A_1 = \begin{pmatrix} 1 & 0 & 1 & 1 \\ -2 & -1 & 2 & 1 \\ 3 & 1 & 0 & 1 \end{pmatrix}, \quad A_2 = \begin{pmatrix} 3 & -3 & 0 & 1 \\ 2 & 3 & 7 & -5 \\ 2 & 3 & -1 & 2 \end{pmatrix} 
$$

a. 确定 $U_1$ 和 $U_2$ 的维数。
b. 确定 $U_1$ 和 $U_2$ 的基。
c. 确定 $U_1 \cap U_2$ 的一个基。

## 练习 2.15
设 $F = \{(x, y, z) \in \mathbb{R}^3 \mid x + y - z = 0\}$ 和 $G = \{(a - b, a + b, a - 3b) \mid a, b \in \mathbb{R}\}$。
a. 证明 $F$ 和 $G$ 是 $\mathbb{R}^3$ 的子空间。
b. 不借助基向量，计算 $F \cap G$。
c. 找出 $F$ 和 $G$ 的基，使用这些基向量计算 $F \cap G$，并验证与上一问题的结果是否一致。

## 练习 2.16
以下映射是否为线性映射？

a. 设 $a, b \in \mathbb{R}$。

$$ 
\Phi: L^1([a, b]) \to \mathbb{R} 
$$

$$ 
f 
\mapsto \Phi(f) = \int_a^b f(x) \, dx 
$$

其中 $L^1([a, b])$ 表示在 $[a, b]$ 上的可积函数集合。

b.

$$ 
\Phi: C^1 \to C^0 
$$

$$ 
f \mapsto \Phi(f) = f' 
$$

其中对于 $k \geq 1$，$C^k$ 表示 $k$ 次连续可微函数集合，$C^0$ 表示连续函数集合。

c.

$$ 
\Phi: \mathbb{R} \to \mathbb{R} 
$$

$$ 
x \mapsto \Phi(x) = \cos(x) 
$$

d.

$$ 
\Phi: \mathbb{R}^3 \to \mathbb{R}^2 
$$

$$ 
x \mapsto \begin{pmatrix} 1 & 2 & 3 \\ 1 & 4 & 3 \end{pmatrix} x 
$$

e. 设 $\theta \in [0, 2\pi)$，

$$ 
\Phi: \mathbb{R}^2 \to \mathbb{R}^2 
$$

$$ 
x \mapsto \begin{pmatrix} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & \cos(\theta) \end{pmatrix} x 
$$

## 练习 2.17
考虑线性映射

$$ 
\Phi: \mathbb{R}^3 \to \mathbb{R}^4 
$$

$$ 
\Phi\left( \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} \right) = \begin{pmatrix} 3x_1 + 2x_2 + x_3 \\ x_1 + x_2 + x_3 \\ x_1 - 3x_2 \\ 2x_1 + 3x_2 + x_3 \end{pmatrix} 
$$

求变换矩阵 $A_\Phi$。确定 $\text{rk}(A_\Phi)$。计算 $\Phi$ 的核和像。$\text{dim}(\text{ker}(\Phi))$ 和 $\text{dim}(\text{Im}(\Phi))$ 分别是多少？

## 练习 2.18
设 $E$ 是一个线性空间。设 $f$ 和 $g$ 是 $E$ 上的两个自同构，使得 $f \circ g = \text{id}_E$（即 $f \circ g$ 是恒等映射 $\text{id}_E$）。证明 $\text{ker}(f) = \text{ker}(g \circ f)$，$\text{Im}(g) = \text{Im}(g \circ f)$，并且 $\text{ker}(f) \cap \text{Im}(g) = \{0_E\}$。

## 练习 2.19
考虑一个内态射 $\Phi: \mathbb{R}^3 \to \mathbb{R}^3$，其变换矩阵（相对于 $\mathbb{R}^3$ 的标准基）为

$$ 
A_\Phi = \begin{pmatrix} 1 & 1 & 0 \\ 1 & -1 & 0 \\ 1 & 1 & 1 \end{pmatrix} 
$$

a. 确定 $\text{ker}(\Phi)$ 和 $\text{Im}(\Phi)$。
b. 确定相对于基

$$ 
B = \left( \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}, \begin{pmatrix} 1 \\ 2 \\ 1 \end{pmatrix}, \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} \right) 
$$

的变换矩阵 $\tilde{A}_\Phi$，即执行向新基 $B$ 的基变换。

## 练习 2.20
考虑 $b_1, b_2, b'_1, b'_2$ 是 $\mathbb{R}^2$ 中的四个向量，它们在 $\mathbb{R}^2$ 的标准基下的表示为

$$ 
b_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}, \quad b_2 = \begin{pmatrix} -1 \\ -1 \end{pmatrix}, \quad b'_1 = \begin{pmatrix} 2 \\ -2 \end{pmatrix}, \quad b'_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix} 
$$

我们定义两个有序基 $B = (b_1, b_2)$ 和 $B' = (b'_1, b'_2)$。
a. 证明 $B$ 和 $B'$ 是 $\mathbb{R}^2$ 的两个基，并绘制这些基向量。
b. 计算从 $B'$ 到 $B$ 的基变换矩阵 $P^1$。
c. 考虑 $\mathbb{R}^3$ 中的三个向量 $c_1, c_2, c_3$，它们在 $\mathbb{R}^3$ 的标准基下的定义为

$$ 
c_1 = \begin{pmatrix} 1 \\ 2 \\ -1 \end{pmatrix}, \quad c_2 = \begin{pmatrix} 0 \\ -1 \\ 2 \end{pmatrix}, \quad c_3 = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix} 
$$

我们定义 $C = (c_1, c_2, c_3)$。
(i) 证明 $C$ 是 $\mathbb{R}^3$ 的一个基，例如通过使用行列式（见第 4.1 节）。
(ii) 设 $C' = (c'_1, c'_2, c'_3)$ 是 $\mathbb{R}^3$ 的标准基。确定从 $C$ 到 $C'$ 的基变换矩阵 $P^2$。
d. 考虑一个同态 $\Phi: \mathbb{R}^2 \to \mathbb{R}^3$，使得

$$ 
\Phi(b_1 + b_2) = c_2 + c_3
$$

$$ 
\Phi(b_1 - b_2) = 2c_1 - c_2 + 3c_3 
$$

其中 $B = (b_1, b_2)$ 和 $C = (c_1, c_2, c_3)$ 分别是 $\mathbb{R}^2$ 和 $\mathbb{R}^3$ 的有序基。确定 $\Phi$ 相对于有序基 $B$ 和 $C$ 的变换矩阵 $A_\Phi$。
e. 确定 $A'$，即 $\Phi$ 相对于基 $B'$ 和 $C'$ 的变换矩阵。
f. 考虑向量 $x \in \mathbb{R}^2$，其在 $B'$ 中的坐标为 $[2, 3]^\top$。换句话说，$x = 2b'_1 + 3b'_2$。
(i) 计算 $x$ 在 $B$ 中的坐标。
(ii) 基于此，计算 $\Phi(x)$ 在 $C$ 中的坐标。
(iii) 然后，将 $\Phi(x)$ 用 $c'_1, c'_2, c'_3$ 表示。
(iv) 使用 $x$ 在 $B'$ 中的表示和矩阵 $A'$ 直接找到这个结果。