
## Eight Queens Puzzle

https://en.wikipedia.org/wiki/Eight_queens_puzzle

Resources used:
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.51.7113&rep=rep1&type=pdf
 * http://gregtrowbridge.com/a-bitwise-solution-to-the-n-queens-problem-in-javascript/
 * https://codereview.stackexchange.com/questions/159946/n-queens-bitwise-approach

### Utility functions (thanks Tom)


```python
import timeit

def timings(fn, maxtime=15, runs=0):
    """Time the given function, fn(N), for increasing values of N, starting from N=6.
    This continues until the calculation time exceeds `maxtime` seconds.
    If a non-zero value for `runs` is given, the function is evaluted that many times
    for each value of N.  If `runs` is zero, we try to pick a number of evaluations
    that totals to roughly 1 second.
    
    A list of (n, timing) tuples is returned.
    """
    result = []
    n, exec_time = 6, 0.001
    while exec_time < maxtime:
        nrun = runs or max(1, int(0.2/exec_time))  # assume time increases ~5x each round.
        exec_time = timeit.timeit("fn(n)", number=nrun, globals=locals()) / nrun
        result.append((n, exec_time))
        n += 1              
    return result
```


```python
import numpy as np

def expfit(n, y):
    """Fit the (n, y) data to a simple exponential model y = exp(a*x + b) and print the results.
    The factor exp(a) and offset b are displayed, along with the predictions for each point.
    
    A list of (n, timing, predicted timing) tuples is also returned.
    """
    cf = np.polyfit(n, np.log(y), 1)
    print("exp(a): {:3f}\nb: {}".format(np.exp(cf[0]), cf[1]))
    yp = np.exp(np.polyval(cf, n))
    print("\n".join(["{:2d}: {:6f}  pred: {:6f}".format(*val) for val in zip(n, y, yp)]))
    return list(zip(n, y, yp))
```


```python
from bokeh.plotting import figure, output_notebook, show
from bokeh.palettes import Category20 
output_notebook()

def plot_timings(*named_timings, log=True):
    """Plot the given timings, each a tuple of the function name, and a matrix of timing data.
    Each matrix row is a 
    """
    args = {'title': "Timings",
            'x_axis_label': 'N',
            'y_axis_label': 'time (sec)'}
    if log:
        args['y_axis_type'] = 'log'

    fig = figure(**args)
    
    # add a line renderer with legend and line thickness
    for i, timing in enumerate(named_timings):
        name, xyp = timing
        if len(xyp[0]) == 3:
            n, y, yp = zip(*xyp)
            fig.line(n, y, legend=name, line_width=2, color=Category20[20][2*i])
            fig.line(n, yp, legend=name + " (pred)", line_width=1, color=Category20[20][2*i+1])
        elif len(xyp[0]) == 2:
            n, y = zip(*xyp)
            fig.line(n, y, legend=name, line_width=2, color=Category20[20][2*i])
        else:
            raise ValueError("Can't decipher timing")

    # show the results
    show(fig)
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="1001">Loading BokehJS ...</span>
    </div>




### Bitwise solution

The bitwise solution I am trying here is based on the information and techniques compiled from different articles.
The idea is to use a set of bit manipulation "tricks" to keep track of the intermediate state of a partially solved board in 3 main bit arrays: "left diagonal", "right diagonal", "column". `1` as a cell value means that we cannot put a queen there, `0` means it is available. 

Important note is that we never keep the whole board in memory approaching the calculation row by row recursively while preparing the next row cell availability as we work on the current row.


```python
# initial implementation
def initial_bitwise(n):
    all_ones = 2 ** n - 1

    def helper(ld, column, rd, count=0):
        if column == all_ones:  # filled out all vacant positions
            return 1

        possible_slots = ~(ld | column | rd)  # mark possible vacant slots as 1s
        while possible_slots & all_ones:
            current_bit = possible_slots & -possible_slots  # get first 1 from the right
            possible_slots -= current_bit  # occupy a slot

            # mark conflicts in the next row
            next_column_bit = column | current_bit
            next_rd_bit = (rd | current_bit) << 1
            next_ld_bit = (ld | current_bit) >> 1

            count += helper(next_ld_bit, next_column_bit, next_rd_bit)
        return count

    return helper(0, 0, 0)
```


```python
initial_bitwise(8)
```




    92




```python
# applying a few suggestions from the codereview.SE
def improved_bitwise(n):
    all_ones = 2 ** n - 1
    count = 0

    def helper(ld, column, rd):
        nonlocal count
        
        if column == all_ones:  # filled out all vacant positions
            count += 1
            return count
        
        possible_slots = ~(ld | column | rd) & all_ones  # mark possible vacant slots as 1s
        while possible_slots:
            current_bit = possible_slots & -possible_slots  # get first 1 from the right
            possible_slots -= current_bit  # occupy a slot

            helper((ld | current_bit) >> 1,
                   column | current_bit,
                   (rd | current_bit) << 1)
        return count

    helper(0, 0, 0)
    return count
```


```python
improved_bitwise(8)
```




    92




```python
time1 = timings(initial_bitwise)
n, y = zip(*time1)
pred1 = expfit(n, y)
```

    exp(a): 4.544137
    b: -18.535040439311505
     6: 0.000118  pred: 0.000079
     7: 0.000378  pred: 0.000357
     8: 0.001388  pred: 0.001622
     9: 0.005692  pred: 0.007369
    10: 0.026315  pred: 0.033485
    11: 0.127757  pred: 0.152160
    12: 0.671756  pred: 0.691435
    13: 3.434624  pred: 3.141973
    14: 19.311666  pred: 14.277557



```python
time2 = timings(improved_bitwise)
n, y = zip(*time2)
pred2 = expfit(n, y)
```

    exp(a): 4.507508
    b: -18.5925765246813
     6: 0.000117  pred: 0.000071
     7: 0.000332  pred: 0.000318
     8: 0.001186  pred: 0.001435
     9: 0.004898  pred: 0.006468
    10: 0.022646  pred: 0.029155
    11: 0.107288  pred: 0.131416
    12: 0.524052  pred: 0.592359
    13: 2.898864  pred: 2.670062
    14: 18.310329  pred: 12.035328



```python
plot_timings(('init', pred1), ('improved', pred2))
```








  <div class="bk-root" id="31c3c16d-04dd-44d7-bab8-8eceb26dab0a" data-root-id="1003"></div>




