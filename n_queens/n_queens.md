
## N Queens Puzzle

https://en.wikipedia.org/wiki/Eight_queens_puzzle

Resources used:
 * https://leetcode.com/problems/n-queens/
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



### Naive Recursive Backtracking solution

The naive backtracking solution below is trying to place a queen into the next cell and backtracks if it reaches a cell where a queen could not be placed.


```python
def naive_backtracking(n):
    count = 0

    def is_safe(grid, row, column):
        nonlocal n

        # check column
        for i in range(column):
            if grid[row][i] == 1:
                return False
        
        # check main diagonal
        for i, j in zip(range(row, -1, -1), range(column, -1, -1)):
            if grid[i][j] == 1:
                return False
        
        # check the other diagonal
        for i, j in zip(range(row, n, 1), range(column, -1, -1)): 
            if grid[i][j] == 1:
                return False
        
        return True
     
    def solve(grid, column):
        nonlocal count, n

        if column >= n:
            count += 1
            return

        for row in range(n):
            if is_safe(grid, row, column):
                grid[row][column] = 1
                solve(grid, column + 1)
                grid[row][column] = 0  # backtrack

    solve([[0 for _ in range(n)] 
            for _ in range(n)], 0)

    return count
        
```


```python
naive_backtracking(8)
```




    92



### Timing & Comparing Solutions


```python
def get_prediction(solution_function):
    timing = timings(solution_function)
    n, y = zip(*timing)
    prediction = expfit(n, y)
    
    return prediction

solution_stats = [
    ('initial_bitwise', get_prediction(initial_bitwise)),
    ('improved_bitwise', get_prediction(improved_bitwise)),
    ('naive_backtracking', get_prediction(naive_backtracking)),
]

plot_timings(*solution_stats)
```

    exp(a): 4.487739
    b: -18.39745518192311
     6: 0.000133  pred: 0.000084
     7: 0.000367  pred: 0.000375
     8: 0.001441  pred: 0.001684
     9: 0.006457  pred: 0.007557
    10: 0.025238  pred: 0.033913
    11: 0.122780  pred: 0.152191
    12: 0.654845  pred: 0.682996
    13: 3.310788  pred: 3.065107
    14: 19.458893  pred: 13.755399
    exp(a): 4.348136
    b: -18.197222372798155
     6: 0.000163  pred: 0.000085
     7: 0.000382  pred: 0.000367
     8: 0.001215  pred: 0.001598
     9: 0.005031  pred: 0.006947
    10: 0.021020  pred: 0.030205
    11: 0.099535  pred: 0.131335
    12: 0.534238  pred: 0.571063
    13: 2.898462  pred: 2.483059
    14: 16.997789  pred: 10.796676
    exp(a): 4.933698
    b: -16.700349101245656
     6: 0.001028  pred: 0.000806
     7: 0.004012  pred: 0.003975
     8: 0.017485  pred: 0.019612
     9: 0.082842  pred: 0.096757
    10: 0.397435  pred: 0.477371
    11: 2.321300  pred: 2.355205
    12: 11.770283  pred: 11.619869
    13: 70.167350  pred: 57.328922









  <div class="bk-root" id="8e6ec7a4-92a5-4813-a6e2-503c1f3dc0d3" data-root-id="1307"></div>




