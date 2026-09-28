# 1. Foundations: teaching a computer with examples

**Start here even if the math looks unfamiliar.** You only need to know how to add, multiply and divide. By the end, you will be able to explain a prediction, measure its mistake and describe one way to improve it. Then try the [gradient descent notebook](../notebooks/01-foundations/gradient_descent.ipynb).

## Step 1: a small shop

Imagine a fruit shop with these receipts. Price is in dollars.

| Apples bought | Total price |
| ------------: | ----------: |
|             1 |           2 |
|             2 |           4 |
|             3 |           6 |

What would you charge for four apples? You probably noticed the rule: **price = 2 times the number of apples**. You used examples (receipts) to discover a rule, then applied it to a new case. That is the basic idea behind _learning from data_. The rule might fail if the shop introduces a bulk discount; examples never guarantee that the future will work exactly like the past.

The _input_ is what we know when making a prediction: the number of apples. The _target_ is the answer we want to learn: the price. In a spreadsheet, each receipt is a **row** (one example), and each measured fact is a **column**. An input column is often called a **feature**. Here there is one feature, `apples`. If we add `delivery_miles`, there are two features. We must not include the total price as an input when trying to predict the total price: that would give away the answer, called **data leakage**.

```python
apples = [1, 2, 3]
prices = [2, 4, 6]
print(2 * apples[1])  # 4, the price for two apples
```

**Pause and predict:** What would the rule say for zero apples? What real-world situation might make the prediction wrong? A delivery fee is one possibility.

## Step 2: give the rule adjustable knobs

Suppose the shop charges a $1 delivery fee plus $2 per apple. We can write "predicted price = (price per apple times apples) + delivery fee." The two adjustable numbers are called **parameters**. We often name them `weight` and `bias`:

$$\text{predicted price}=\text{weight}\times\text{apples}+\text{bias}.$$

For four apples, weight 2 and bias 1, this predicts $2\times4+1=9$. Only now do we introduce shorthand: $x$ means input, $w$ weight, $b$ bias, and $\hat y$ (read "y-hat") predicted answer. Thus $\hat y=wx+b$. The actual answer is $y$. The names are not magic: they shorten the sentence above.

For many examples, people store inputs in an array called $X$. Its **shape** is simply its size in each direction. Three receipts with two input columns make a $3$-row, $2$-column table, written $[3,2]$. Multiplying each row's two features by two weights and adding them gives three predictions. This multiply-and-add is a **dot product**. For features `[2, 3]`, weights `[4, -1]` and bias `1`, the prediction is $2\times4+3\times(-1)+1=6$. A Python array library such as NumPy handles many rows at once; check `.shape` whenever an operation gives a surprising result.

## Step 3: count mistakes

Pretend the true price for two apples is $5$, but our current rule has weight $0$ and bias $0$. Our prediction is $0$; the **error** (prediction minus truth) is $0-5=-5$. To give both a $5$ undercharge and a $5$ overcharge a positive cost, square the error: $(-5)^2=25$. This cost is called a **loss**. If there are many receipts, average their squared errors: this is _mean squared error_ (MSE). Small MSE means closer predictions **on these examples**, not necessarily on unseen customers.

## Step 4: improve the knob, one move at a time

Keep the delivery fee at zero and use only the two-apple receipt. Our rule is $\hat y=w\times2$. The current weight is zero. The slope of the squared-error loss at this point is $-20$: increasing the weight a little makes the loss smaller. Choose a **learning rate**, or step size, of $0.1$. A gradient-descent update subtracts step size times slope:

$$w_{\text{new}}=0-0.1\times(-20)=2.$$

Now the prediction is $2\times2=4$, error $4-5=-1$, and loss $1$. We improved from 25 to 1 in one step. Try it with arithmetic before using the notebook. If the learning rate were $1$, the new weight would be $20$, the prediction $40$ and the loss $(40-5)^2=1225$: an overlarge step made things worse.

The word **gradient** means the slope when there are multiple adjustable knobs. A derivative measures how a small change to one knob changes the loss. The **chain rule** lets us work through a sequence of operations, such as "multiply, then subtract, then square." For this one example, loss $(2w-5)^2$ has derivative $2(2w-5)\times2$, which is $-20$ at $w=0$. You do not have to calculate every derivative by hand in later chapters; PyTorch can do that, but this tiny case explains what it is calculating.

## Step 5: think about uncertain answers

For prices, we predict a number. For "will a customer buy?", we might predict a probability such as 0.8 (80 out of 100 similar cases). **Calibration** asks whether predictions near 0.8 really succeed roughly 80% of the time. A model can produce a number between zero and one without being calibrated.

Base rates matter. Imagine a screening test used on 1,000 people. Only 10 actually have a condition (1%). A test finds 9 of those 10, but incorrectly flags 99 of the 990 others. Among the 108 flagged people, only 9 have the condition: $9/108\approx8.3\%$. Even a test with good-sounding accuracy can give many false alarms when the condition is rare. **Bayes' rule** is the general way to update a prior chance using new evidence:

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.$$

Here $P(A)$ is the chance before seeing a result and $P(A\mid B)$ is the chance after. You can return to the formula later; the 9-out-of-108 count is the intuition. **Mean** is an average, **variance** tells us how spread out values are, and **correlation** tells us whether quantities move together, not whether one caused the other. Bernoulli refers to one yes/no outcome, binomial to a count of yes outcomes, and Gaussian to a bell-shaped continuous distribution; none is a universal description of real data.

## Try it yourself

1. A shop charges $3$ per apple plus $2$ delivery. What is the prediction for four apples? **Answer:** $3\times4+2=14$.
2. True price is $5$ and prediction is $3$. What is the squared-error loss? **Answer:** $(3-5)^2=4$.
3. Why is including tomorrow's receipt total when predicting today's sale a problem? **Answer:** that information does not exist when the prediction must be made; it leaks the answer.
4. For 100 rows and five feature columns, what does $[100,5]$ describe? **Answer:** 100 examples, each measured in five ways. Leave matrix multiplication for later.

**Next:** run the [notebook](../notebooks/01-foundations/gradient_descent.ipynb), then learn [how classical AI searches rather than learns](02-classical-ai.md) and [how to test a learned rule](03-machine-learning.md). For a second treatment of tensors and optimization see [PyTorch's beginner sequence](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
