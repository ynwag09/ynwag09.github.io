---
layout: posts
title: Boosting
tags : boosting adaboost gbm
use_math: true
---

# Boosting

## 1. Idea

---

Combine successive *simple Week learners* which produce powerful committee 

$$
G(x) = \sum_{m=1}^{M} \alpha_{m} G_{m}(x)
$$

Originally desgined for classification, but since it's 'meta-algorithm', can be used also for Regression and ranking. 

많은 종류의 boosting 기법들간의 차이는 학습데이터의 가중치 부여 방식과 loss function의 정의이다.


$$
\underset{\beta_{m}, \gamma_{m}}{\operatorname{argmin}}\sum_{i=1}^{N}L(y_{i}, \sum_{m=1}^{M} \beta_{m} G_{m}(x_{i} ; \gamma_{m}))
$$

부스팅의 fitting은 위의 식을 푸는게 일반적인 방법이지만, forward stagewise additive modeling 을 통해

$$
\underset{\beta, \gamma}{\operatorname{min}}\sum_{i=1}^{N}L(y_{i}, \beta G(x_{i};\gamma)) 
$$
의 해로 근사할수있다.


## 2. Adaboost

### 1) binary classification 
---

![adaboost_framework](../images/adaboost_framework.png)

Adaboost.M1 은 *exponential loss* 을 사용하는 forward stagewise additive modeling 과 같다. 증명은 밑에서

week learner $G_{m}(x)$의 output은 $\{0,1\}$ 이다.

$$
\begin{aligned}
(\beta_m, G_m) &= \underset{\beta, G}{\operatorname{argmin}}\sum_{i=1}^{N}\exp(-y(f_{m-1}(x_i) + \beta G(x_i))) \\
&=\underset{\beta, G}{\operatorname{argmin}}\sum_{i=1}^{N} w_i^{(m)} \exp(-\beta y G(x_i)) -(1)   \\
& \quad w_i^{(m)} = \exp(-y f_{(m-1)}(x))\\
\end{aligned} 
$$

의 해는 $G_m$ 을 찾은후 $(1)$ 에 대입해 $\beta_m$을 구한다.

$$ G_m = \underset{G}{\operatorname{argmin}}\sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i))
$$
$(1)$ 의 식은
$$
\begin{aligned}
\exp(-\beta) \cdot \sum_{y_i = G(x)}w_i^{(m)} + \exp(\beta) \cdot \sum_{y_i \not= G(x)}w_i^{(m)}\quad\quad\quad\quad\\
\exp(-\beta) \cdot \sum_{i=1}^{N}w_i^{(m)} + (\exp(\beta) - \exp(-\beta)) \cdot \sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i)) 
\end{aligned} 
$$
을 $\beta$에 대해 미분하면
$$
\begin{aligned}
&(\exp(\beta) + \exp(-\beta)) \cdot \sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i)) =\exp(-\beta) \cdot \sum_{i=1}^{N}w_i^{(m)} \\
&\exp(\beta)  \cdot \sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i))  = \exp(-\beta) \cdot (\sum_{i=1}^{N}w_i^{(m)} - \sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i)))\\
&\exp(2\beta)  \cdot \sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i))  =(\sum_{i=1}^{N}w_i^{(m)} - \sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i)))\\
&\exp(2\beta) =\frac{(\sum_{i=1}^{N}w_i^{(m)} - \sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i)))}{\sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i))}\\
&let \ \  err_m = \frac{\sum_{i=1}^{N}w_i^{(m)}I(y_i \not= G(x_i))}{\sum_{i=1}^{N}w_i^{(m)}}\\
&\beta = \frac{1}{2} \cdot \log \frac{1 - err_m}{err_m}
\end{aligned}
$$
update 되는 training weight는 $w_{i}^{(m+1)} = w_{i}^{(m)} \exp(- \beta \ I(y_i \not =G_m(x_i))$이다.

$err_m < \frac{1}{2}$ 이 assume 된다.    (better than random guessing)
$log (\frac{1-x}{x})$ 의 그래프를 생각하면 $\beta_m$은 항상 양의 값을 갖는다.  
즉 weight가 지속적으로 조정되면서 additive하게 모델이 바뀐다는 것이다.

### 2) multiclass classification

Multi-class AdaBoost(2009) - Ji Zhu, Hui Zou, Saharon Rosset and Trevor Hastie  
의 내용을 발췌해 정리했다.

1) Background

 - Purpose 
    - 기존 adaboost.m 알고리즘(forward stagewise additive modeling)을 활용해 multi class classifer을 class-reduce 없이 만듬
    - exponential loss 가 fisher-consistent loss function for multi-class classification 의 memeber임을 보임


 - Limitation of adaboost.M
    - $err_m$의 전제가 0.5보다 작은반면  
     multi-class에서의 random guessing의 prob 은 $\frac{k-1}{k}$ 보다만 작으면 된다.

    - 이것은 그림1 에서 볼수 있듯이 multi-class task 에서 $err_m$ 이 $\frac{1}{2}$에서 벗어나지 못하고 (weight가 변하지 않음) model이 update되지 못함을 보여준다.

<figure>
    <img
    src="../images/limit_ada.png">
</figure>


2) SAMME as forward stage additive modeling

<figure>
    <img
    src="../images/samme.png">
</figure>

  - Settings
    - output c with K dim y vector
$$
        y_k =
        \begin{cases}
            1, & \text{if}\ c = k \\
            -\frac{1}{K-1}, &\text{if}\  c \not=k
        \end{cases}
        $$
        $\text{if} \ K = 4 , c = 3, y = (-\frac{1}{3}, \ \frac{1}{3}, \ 1, \ \frac{1}{3})$  
    
    - want to find $\mathbf{f}
    (x) = (f_1(x),...,f_k(x))$ such that
    $$
    \underset{\mathbf{f}(x)}{\operatorname{min}} \ \sum_{i=1}^{N} L(y_i, f(x_i)) \quad \text{subject to} \ f_1(x) + ... + f_k(x) = 0 
    $$
    f follows that
    $
    \mathbf{f}(x) = \sum_{k=1}^{K} \beta_k G_m(x) 
    $


​    
  - estimation

    $$
    \begin{aligned}
    \underset{\mathbf{f}}{\operatorname{argmin}} \ E_{Y|x}(\exp(-\frac{1}{K}(Y^Tf)) \quad \text{subject to }\ f_1(x) + ... + f_k(x) = 0  
    \end{aligned}
    $$

    $$
    \begin{aligned}
    E_{Y|x}(\exp(-\frac{1}{K}(Y^Tf))  &= \sum_{k=1}^{K} P(c = i | x) \exp(-\frac{1}{K}(y_1f_1+y_2f_2+\cdots + y_kf_k)) \\
    &=\sum_{k=1}^{K} P(c = i | x) \exp(-\frac{1}{K}(-\frac{1}{K-1}f_1+-\frac{1}{K-1}f_2+\cdots + f_i(x)+\cdots + -\frac{1}{K-1}f_k))\\
    &= \sum_{k=1}^{K} P(c = i | x) \exp(-\frac{1}{K}(-\frac{1}{K-1}\sum_{k=1}^{K}f_i(x)+ \frac{K}{K-1} f_i(x)))\\
    &= \sum_{k=1}^{K} P(c = i | x) \exp(-\frac{1}{K-1} f_i(x)))
    \end{aligned}
    $$

    라그랑주 승수법을 이용해 제약식을 풀면 

    $$
    \begin{aligned}
    \mathcal{L}(f,\lambda) &= \sum_{k=1}^{K} P(c = i | x) \exp(-\frac{1}{K-1} f_i(x))) - \lambda(f_1(x) + ... + f_k(x))\\
        \frac {\partial \mathcal{L}}{\partial f} &= -\frac{1}{K-1} P(c = i | x) \exp(-\frac{1}{K-1} f_i(x))) - \lambda = 0  \quad \quad \ \ - (1) \\
        \frac {\partial \mathcal{L}}{\partial \lambda} &= -(f_1(x) + ... + f_k(x)) = 0
    \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \ \ - (2)\\
    \end{aligned}
    $$
    from (1)
    $$
    \begin{aligned}
    &\exp(-\frac{1}{K-1}f_i(x)) = - \frac{(K-1) \lambda}{P(c = i | x)} \\
    &f_i(x) = (K-1)\bigg(log{P(c=i|x)} - log{(-(K-1)\lambda)}\bigg) -(3)
    \end{aligned}
    $$
    (3)을 (2)에 대입하고 이를 다시 (3) 에 대입한다.

    $$
    \begin{aligned}
    &\sum_{k=1}^{K}log{P(c=k|x)} = K log{(-(K-1)\lambda)}\\
    &f_i(x) = (K-1)log{P(c=i|x)} - \frac{K-1}{K} \sum_{k=1}^{K}log{P(c=k|x)} \\
    &\underset{k}{\operatorname{argmax}f_k(x)} = \underset{k}{\operatorname{argmax}}Prob(c = k | x)
    \end{aligned}
    $$
    which follows bayes classification rule

---

- marginal estimation of $f$
    $$
    \begin{aligned}
    (\beta_m, G_m) &= \underset{\beta, G}{\operatorname{argmin}}\sum_{i=1}^{N}\exp(-\frac{1}{K}y_i^{T}(f_{m-1}(x_i) + \beta_m G_m(x_i))) \\
    &= \underset{\beta, G}{\operatorname{argmin}}\sum_{i=1}^{N}w_i \exp(-\frac{1}{K} \beta_my_i^{T} G_m(x_i)) \quad \quad \quad -(4) \\
    & \text{where} \quad w_i = \exp(-\frac{1}{K}y_i^{T}f_{m-1}(x_i)) 
    \end{aligned}
    $$

    every G(x) has one-to-one correspondence with multi-class classifier T(X) with following
    $$
    \begin{aligned}
    T(x) &= k , \text {if} \ \ g_{k} =1\\
    g_k(x) &=         \begin{cases}
            1, & \text{if}\ T(x) = k \\
            -\frac{1}{K-1}, &\text{if}\  T(x) \not=k
        \end{cases}
    \end{aligned}
    $$
    then,
    $$
    \begin{aligned}
    T^{(m)}(x) =  \underset{T}{\operatorname{argmin}}\sum_{i=1}^{N} I(c_i \not= T(x_i))
    \end{aligned}
    $$

    이를 (4)에 넣어 정리하면
    $$
    \begin{aligned}
    &\sum_{c_i = T(x_i)} w_i \exp(-\frac{1}{K-1}\beta) + \sum_{c_i \not= T(x_i)} w_i \exp(\frac{1}{(K-1)^2}\beta) \\ 
    &\exp(-\frac{\beta}{K-1})\sum_{i=1}^{N}w_i + \bigg(\exp(\frac{\beta}{(K-1)^2})-\exp(-\frac{\beta}{K-1})\bigg)\sum_{i=1}^{N}w_i I(c_i \not= T(x_i))
    \end{aligned}
    $$
    이를 $\beta$에 대해 미분하면

    $$
    \begin{aligned}
        \frac{\partial{}}{\partial{\beta}} & = - \frac{1}{K-1} \exp(-\frac{\beta}{K-1})(\sum_{i = 1}^{N} w_i) + \bigg(exp(\frac{\beta}{(K-1)^2}) - exp(-\frac{\beta}{K-1}) \bigg)\sum_{i=1}^{N}w_i I(c_i \not= T(x_i)) = 0 \\
    \end{aligned}
    $$
    이고 이를 정리하면
    $$
    \begin{aligned}
    \beta &= \frac{(K-1)^2}{K}\bigg(log{\frac{1-err_m}{err_m}} + log{(K-1)} \bigg) \\ 
    &\text {where} \quad err_m = \frac{\sum_{i=1}^{N}w_i I(c_i \not= T(x_i))}{\sum_{i = 1}^{N}w_i}  
    \end{aligned}
    $$


### 3) regression 

Drucker, “Improving Regressors using Boosting Techniques”, 1997.  
의 내용을 발췌해 정리했다

분석 방법은 adaboost.m과 동일하고 loss function과 weight update 방식을 확인한다

$ L_i = L\bigg[|y_i^{(p)}(x_i) - y_i| \bigg] ,  \quad D = sup|y_i^{(p)}(x_i) - y_i| $

then ,

$$
\begin{aligned}
    L_i &= \frac{|y_i^{(p)}(x_i) - y_i|}{D} \quad \text {linear} \\
    L_i &= \frac{(|y_i^{(p)}(x_i) - y_i|)^2}{D} \quad \text {squared}\\
    L_i &= 1 - \exp(\frac{|y_i^{(p)}(x_i) - y_i|}{D}) \quad \text {exponential}    
\end{aligned}
$$

$ \text {average loss} : \bar{L} = \sum_{i=1}^{N} L_i p_i $
measure of confidence in predictor  $\beta = \frac{\bar{L}}{1-\bar{L}}$

weight update $w_i = w_i \beta^(1-\bar{L}) $
### 4) robustness

loss function이 outlier에 얼마나 민감한가에 따라 robustness가 다르다

adaboost에서 일반적으로 사용하는 exponential loss 와 squared loss는 상대적으로 계산에 용의 하지만 outlier에 민감하다.

<figure>
    <img
    src="../images/loss_robust.png">
</figure>

## 3. GBM

## 4. regularization

boosting 계열의 문제중 하나는 overfitting 이다.

이를 해결하기 위해 size of tree, learning rate(proportion of new model), number of iteration(early stopping)를 적절히 선택해야한다.

 - shrinkage (v)
   - number of iteration(M) 과 trade off 관계가 있다
   - Empirically v< 0.1 보다 작게 하고 M을 관찰에 early stop하는게 performance가 좋고 test error 가 저점에서 오래 유지된다.

 - subsampling
   - boosting은 가중치의 변화만 줄뿐 모든 learner의 X는 동일하다. 
     generalization을 위해 일부 portion만 sampling해 분석을 반복한다.
   - subsampling 단독으로만 사용되면 performace가 좋지 않다. 다른 regularization과 함께 사용

## 5. interpretation

  - relative importance

  