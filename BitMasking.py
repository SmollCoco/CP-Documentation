wA = 69; i = 3; n = 3 #Examples
# Consider 0-indexed representation for the following with 0 being the far right digit/element (LSB), thus :
# i ranges from 0 to len(A) - 1:

# ---------------------------------- BASIC MASKING MANIPULATION ------------------------
# Check if i'th bit is set of A: 
def isON(A,i):                                                                          
    return A & (1 << i)
#Set i´th bit to ON :
def setON(A,i):
    A |= (1 << i)

#Clear i'th bit ( turn OFF ):
def clear(A,i):
    A &= A & ~(1 << i)

# toggling (switching ON/OFF) i'th bit:
def toggle(A,i):
    A ^= (1 << i)

# ---------------------------------- DIVISON -------------------------------------------
#Dividing/Multiplying by 2^n:

A / (2**n) == A >> n
A * (2**n) == A << n

#Check if 2**n divides A:

def isDivsible2powern(A,n):
  for i in range(n):
    if A & (1 << i):
      return False
  return True

# ---------------------------------- GRAY CODE SEQUENCE --------------------------------

# i'th element in the Grey Code Sequence :

def greycode(k):
 return k ^ (k>>1)

#reverse Gray-Code: given a gray-code g, find it's position in the GrayCode Sequence:

def rev_g(g):
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n

# ---------------------------------- SUBMASK ENUMERATION --------------------------------

# Iterating through submask ( iterate through only set elements):
    #consider mask m:
    #s iterating through submasks: meaning s would only take the value of ON elements of mask m :
    #Example : for m = 0b1100 = 12 , s will iterate through m = 0b1100 = 12 -> 0b1000 = 8 -> 0b100 = 4 --> 0
m = 12 #example

s = m
while s > 0:       
    #procedure
    s = (s-1) & m 

#--------------------------------BRIAN KERNIGHAN'S ALGO----------------------------------
#Clear lowest set bit:
# example : 1011000 ---> 1010000
def clearMostRight(n):
    n &= (n-1)

#Brian Kernighan's algorithm : counting the number of set bits
# Each time we clear the lowest set bit, we add 1 to the counter
# We keep doing so until n == 0, time complexity : O(log(n)) (worst case)

def numofsetbits(n):
    count = 0
    while n > 0:
        n &= n-1
        count += 1
    return count

#Counting amount of set bits of all number from 0 -> n : 
#Using the Brian-Kernighan's algorithm would result in a time complexity of O(nlog(n))
#But we can use a much simpler approach
#For numbers from 0 to (2^x) - 1, we have x * 2^(x-1) set bits, thus :
# We count number of bits x in n, we add x * 2^(x-1) to count, we count the number of set bits between n and (2^x)-1 and add it to count
# we repeat for n - ( 2^(x)-1 )

def countSetBits(n):
    count = 0
    while (n > 0):
        x = len(bin(n)[2:]) - 1
        count += x << (x - 1)
        n -= 1 << x
        count += n + 1
    return count

#------------------------------------------- GENERATE ALL POSSIBLE SUBSETS-----------------------------------

#Find all subsets of a given set ( create superset ):
# time complexity : O(n * 2^n)

def createPowerset(arr,n):
    pset = []
    for i in range(2**n):
        subset = []

        for j in range(n):
            if i & (1 << j):
                subset.append(arr[j])
        pset.append(subset)
    return pset       
