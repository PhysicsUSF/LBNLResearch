import sys
import cStringIO
def test():
    '''
    Purpose: To show how to suppress the print statements in a function for a doctest.
    To run:
    
    python test2.py -v
    

    doctest:
    
    (modified from:http://stackoverflow.com/questions/9949633/suppressing-print-as-stdout-python
    also take a look at this (the decorator method seems pretty elegant:
    http://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    
    >>> actualstdout = sys.stdout
    >>> sys.stdout = cStringIO.StringIO()
    >>> result = test()
    >>> sys.stdout = actualstdout
    >>> sys.stdout.write(str(result))
    4.0
    '''

    # note: when one runs doctest, results of the print statements don't show at stdout.
    print 2.
    x = 3.4
    print x
    
    return 4.



if __name__ == "__main__":
    import doctest
    doctest.testmod()