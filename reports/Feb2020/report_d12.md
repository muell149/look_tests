# Algorithm on Face Recognition

Based on the <cite>[paper][1]</cite> by Hongun Li, we are going to use a relatively novel approach for Face Recognition. 

The main algorithm is as follows:

### Input: 
First, we have the training data $X=\{x_1,x_2,\dots,x_q\}$ and the testing data $Y=\{y_1,y_2,\dots,y_p\}$. The size of training and testing data is $S_1\times S_2$. The parameters are $\lambda$, $\rho$. A matrix of normalized training samples $D=[D_1,D_2,\dots,D_k]\in R^{m\times n}$ for $K$ classes, and a test sample $y\in R^m$. As you can see, the dimension $m$ of the training sample is the same as the dimension of each $D_i$, and the dimension $n$ denotes the number of images on each class.

### Algorithm:

* Make the Dynamic Rank estimation and get the optimal rank $r$.
* For each class $i$ do 
    * Obtain the low-rank part  $M_{i}$ of the training data $D_i$. 

        In order to do this, first we need to conduct a SVD of the matrix $M_i=U\Sigma V^T$, then, having obtained the matrix $\Sigma$, where its diagonal are the ordered singular values, take just the first $r$ singular values and set the remaining to zero, this will give a new matrix $\Sigma'$. 
        
        Finally $M_{i}=U\Sigma' V^T$.
    * Given $M_{i}$, solve the following optimization problem

        <p><br>

        $$\min_{Z_i^{*},L_i^{*},E_i^{*}} ||Z_i||_{*} + \lambda ||M_i-M_i Z_i||_F^2$$
        <p><br>

        $$s.t.\hspace{1cm} M_i=M_iZ_i+L_iM_i+E_i,\hspace{0.5cm}Z_i=V_{M_i}W_{Z_i}V_{M_i}^T,$$

        <p><br>

        $$ \hspace{1.5cm}L_i=U_{M_i}\left(I-W_{Z_i}\right)U^T_{M_i},\hspace{0.5cm}\text{rank}(Z_i)=r$$ 

        <p><br>

        As you can see, the variables we are looking are $W_{Z_i}$ and $E_i$, in order to find the best $Z_i$ and $L_i$. The other components can be found by conducting a skinny SVD of $M_i=U_{M_i}\Sigma_{M_i} V^T_{M_i}$ to find $U_{M_i}$ and $V_{M_i}$. The meaning of skinny is that $\Sigma_{M_i}$ is a square matrix of size $\text{rank}(M_i)$.

        This is a kind of optimization problem called convex optimization problem (you can read more about it in the following <cite>[link][2]</cite>), where $||Z_i||_{*} + \lambda ||M_i-M_i Z_i||_F^2$ is called the "objective function", there are no inequality constraints and there are four equality constraint functions. It is called convex because the objective and constraint functions satisfy 


        $$f(\alpha \vec{x}+\beta\vec{y})\leq \alpha f(\vec{x})+\beta f(\vec{y})$$

        for all $\vec{x},\vec{y}\in R^n$, and all $\alpha,\beta\in R$, with $\alpha + \beta = 1$, $\alpha\geq 0$, $\beta \geq 0$.
        
        Every norm satisfies this inequality.

        There are different approaches to solve this kind of problems, and there is a Python library created called <cite>[CVXOPT][3]</cite>.
    * Having got the optimal $L_i$, calculate the discriminative information of face image by $L_iM_i$.
    * For the identification, calculate the residual associated with the ith class as

        $$e_i(y)=||L_iy-L_iM_i||^2_2$$

        Note that this will be a function of $y$, so that the identification is given by

        $$\text{identify}(y)=\text{argmin}_i(e_i(y))$$

        i.e. the class $i$ where the function $e_i$ is minimized


[1]:https://doi.org/10.1016/j.patcog.2016.05.014
[2]:https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
[3]:https://cvxopt.org/index.html