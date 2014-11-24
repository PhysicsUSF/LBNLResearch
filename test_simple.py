import sys
import cStringIO
def test():
    '''
    >>> actualstdout = sys.stdout
    >>> sys.stdout = cStringIO.StringIO()
    >>> result = test()
    >>> sys.stdout = actualstdout
    >>> sys.stdout.write(str(result)+'\n')
    >>> sys.stdout.flush()
    >>> sys.exit(0)
    4.0
    '''

    print 2.
    return 4.



if __name__ == "__main__":
    import doctest
    doctest.testmod()