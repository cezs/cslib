{-# LANGUAGE MultiParamTypeClasses, TypeFamilies #-}
{-|
Module: BasicsCollection
Description: Initial Attempts
License: -
Maintainer: c.stankiewicz@wlv.ac.uk

First module written in Haskell language (# items)

Use GHCi with @:load BasicsCollection@.
To reload use @:reload@.

Also, see $todos$.
-}
module BasicsCollection where

import Data.Char
import Data.Set     hiding (map)
import Data.IntSet  hiding (map)
import Prelude hiding ((==), (/=))
import Data.List as List

--------------------------------------------------------------------------------
-- functions -------------------------------------------------------------------
--------------------------------------------------------------------------------

{-|-}
laugh = "Ha!"

{-|-}
sq x = x * x

{-|-}
addOne n = 1 + n

{-|-}
volbox x y z  = x * y * z

{-|-}
fsin a r = sin(a) * r

{-|-}
powEach x = [1..10]**x

{-|-}
f = (+3)

{-|-}
g foo x = foo x * foo x

{-|-}
g2 :: Num a => (t -> a) -> (t -> a) -> t -> a
g2 f1 f2 x = f1 x + f2 x

{-|-}
multThree x y z = x * y * z

{-|-}
multWithNine = multThree 9

{-| reverse sequence given with boundaries -}
farrayRev a z = reverse [a .. z]

{-|-}
fact :: Integer -> Integer
fact n = product [1..n]

{-|-}
addVec a b = (fst a + fst b, snd a + snd b)

-- if-then-else ----------------------------------------------------------------

{-|-}
pows n m = if m > 0 then m - 1 else n * n

{-|-}
doubleSmallNumber x = if x > 1 then x else x + x

{-|-}
dsn x = (if x > 100 then x else x * 2) + 1

-- where -----------------------------------------------------------------------

{-|-}
foo n = n * pi * s / c
    where s = n * 2
          c = n * 4

{-| multiply number by it's cube -}
multByCube :: Int -> Int
multByCube x  = x * i where i = x^3

{-|-}
volcyl r y = areac * y  where areac = sq (pi  * r)

{-|-}
areat a b c = sqrt ( s * (s - a) * (s - b) * (s - c) ) where s = (a + b + c) / 2
	
{-|-}
numofs a b
    | num > 0 = "pos"
    | num < 0 = "neg"
    | otherwise = "neu"
        where
        num = a + b

{-|-}
qroots (a, b, c) =
  if d < 0 then error "confused"
    else (r1, r2)
      where r1 = e + sqrt d / (2 * a)
            r2 = e - sqrt d / (2 * a)
            d = b * b - 4 * a * c
            e = -b / (2 * a)

-- let-in ----------------------------------------------------------------------

{-|-}
foo2 n = let a = 4 / 5
             cube = n ^ 3
         in a * pi * cube

{-|-}
cylinder :: (RealFloat a) => a -> a -> a  
cylinder r h = 
  let sideArea = 2 * pi * r * h
      topArea = pi * r ^2
  in  sideArea + 2 * topArea  

-- case-of ---------------------------------------------------------------------

{-| example of 'case' and 'of' -}
classify :: (Num a, Eq a) => a -> [Char]
classify age = case age of 0 -> "newborn"
                           1 -> "infant"
                           2 -> "toddler"
                           _ -> "..."

-- guards -----------------------------------------------------------------------

maxi :: Int -> Int -> Int
maxi a b | a > b = a
         | otherwise = b

{-| highest of the 4 Int arguments -}
maxed :: Int -> Int -> Int -> Int -> Int
maxed a b c d = maxi a (maxi b (maxi c d))

{-| lowest of the 2 Int arguments -}
lowest :: Int -> Int -> Int
lowest a b | a < b = a
           | otherwise = b 


{-| maximum of the [Int] -}
maxList ::  [Int] -> Int
maxList []  = 0
maxList (a:az)
   | maxList az > a = maxList az
   | otherwise = a

{-| guarded recursion -}
recuu _ [] = []
recuu n _
    | n <= 0 = []
recuu n (x:xs) = n * x : recuu n xs

{-| guarded recursion -}
recuuu _ [] = []
recuuu n _
    | n <= 0 = []
recuuu n (x:xs) = x : recuuu (n-1) xs

factorial :: (Integral a) => a -> a
factorial n | n < 2 = 1
factorial n = n * factorial (n - 1)

myMax a b c
  | c > d = c
  | otherwise = d
  where d = a * b


-- recursion --------------------------------------------------------------------

{-|-}
listlength :: [Integer] -> Integer
listlength [] = 0
listlength (x:xs) = 1 + listlength xs

{-|-}
recu [] = []
recu (x:xs) = x * x : recu xs

{-|-}
infinity :: a -> [a]
infinity x = x : infinity x

{-|-}
take' x n = recuuu x (infinity n)

{-|-}
maksimum :: (Ord a, Num a) => [a] -> a
maksimum [] = 0
maksimum (current_maksimum : container_has)
         | maksimum container_has > current_maksimum
           = maksimum container_has
             | otherwise = current_maksimum

{-| take n elems from xs -}
takeElems _ [] = []
takeElems n _
	| n <= 0 = []
takeElems n (x:xs) = x : takeElems (n-1) xs

{-|-}
gDown :: Integer -> [Integer]
gDown (-5) = [-5]
gDown n = n : gDown (n - 1)

{-|-}
myDrop :: Int -> [a] -> [a]
myDrop n xs = if n <= 0 || Prelude.null xs then xs else myDrop (n - 1) (tail xs)

{-|
example of usage:

  >>> quicksort [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]

  [1,2,3,4,5,6,7,8,9,10]
-}
quicksort :: (Ord a) => [a] -> [a]  
quicksort [] = []  
quicksort (x:xs) =
  let smallerSorted = quicksort [a | a <- xs, a <= x]
      biggerSorted = quicksort [a | a <- xs, a > x]
  in  smallerSorted ++ [x] ++ biggerSorted

-- pattern matching -------------------------------------------------------------

{-| Get 1st element -}
st :: (a, b, c) -> a
st (x , _, _) = x

{-| Get 2nd element -}
nd :: (a, b, c) -> b
nd (_ , y, _) = y

{-| Get 3rd element -}
rd :: (a, b, c) -> c
rd (_ , _, z) = z

{-| Can precede tail of a list with as many elems as wished -}
{-| Note: xs needs at least 4 elements -}
lLen [] = 0
lLen (x:y:z:o:xs) =  1 + lLen xs

{-| Note: xs needs at least 4 elements -}
asum :: [Integer] -> [Integer]
asum [] = []
asum (x:y:z:o:xs) = x + x : asum xs

{-|-}
multiplyList :: Integer -> [Integer] -> [Integer]
multiplyList _ [] = []
multiplyList m (n:ns) = (multiplyByM n) : multiplyList m ns
    where
    multiplyByM x = m * x

{-|-}
fac::(Num a, Eq a) => a -> a
fac 0 = 1
fac n = n * fac(n-1)

-- list comprehesion -----------------------------------------------------------

lc xs = [foo | (n, m) <- xs, let foo = n + m, foo >= 10]

-- rightTriangles = [(a,b,c) | c <- [1..128], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2 ]

myConc przym rzecz = [prz ++ " " ++ rz | prz <- przym, rz <- rzecz]

calcBmis :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis xs = [bmi w h | (w, h) <- xs]
  where bmi weight height = weight / height ^ 2  

calcBmis2 :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis2 xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2, bmi >= 25.0]  

-- functorial -----------------------------------------------------------------
          
{-| dump shift of sequence provided with boundary paramaters -}
farrayRemap mov f t = map (+mov) [f .. t]

{-|-}
-- p = filt_pinP [2..]
--   where filt_pinP (p:xs) = 
--           p : filt_pinP [x | x <- xs, x `mod` p /= 0]

-- monadic values --------------------------------------------------------------

{-|-}
myfun [] = Nothing
myfun a = Just a

{-|-}
mylookup _ [] = Nothing
mylookup key ((x,y):s) = if key == x then Just x else mylookup key s

--------------------------------------------------------------------------------
-- data, type, class, instance -------------------------------------------------
--------------------------------------------------------------------------------

-- Error: Illegal literal in type (use DataKinds to enable): 0
-- data Binr = 0 | 1

{-|-}
data NCTree a = NCNode a (NCTree a) (NCTree a) | Empty deriving (Show)

{-|-}
data Shp = Shp1 Float Float Float | Shp2 Float Float Float Float

{-|-}
surf :: Shp -> Float
surf (Shp1 _ _ r) = pi * r * r
surf (Shp2 c1 c2 b1 b2) = (b1 - c1) * (b2 - c2)

{-|-}
data SomeStru = SomeStru { entr1 :: String, entr2 :: Int } deriving (Show)

{-|
Possible use:
@
instance Cp Dire where
    Ent1 == Ent1 = True
    Ent2 == Ent2 = True
    Ent3 == Ent3 = True
    _ == _ = False
@
-}
class Cp a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x == y = not ( x /= y)

type Endomorphic a = a -> a

-- itsme :: (Endomorphic a) => a -> a
-- itsme a = a

{-|-}
class Equal a where
--  data TT a :: *->*
  eq, neq :: a -> a -> Bool

{-|-}
instance Equal Bool where
  true `eq` b = b
  false `eq` b = not b
  c `neq` b = not (c `eq` b)

{-|-}
class Collection c where
  type Element c :: *
  cmember::Element c->c->Bool

{-|-}
instance Ord a => Collection (Set a) where
  type Element (Set a) = a
  cmember = Data.Set.member

{-|-}
type Id = Int

{-|-}
type Name = String

{-|-}
data Stru = Id Name 

data Car =
  Car {
    company :: String,  -- ^ Company record field
    model :: String,  -- ^ Model record field
    year :: Int  -- ^ Year record field
    } deriving (Show)

data Day
  = Monday  -- ^ Monday constructor
  | Tuesday  -- ^ Tuesday constructor
  | Wednesday  -- ^ Wednesday constructor
  | Thursday  -- ^ Thursday constructor
  | Friday  -- ^ Friday constructor
  | Saturday  -- ^ Saturday constructor
  | Sunday  -- ^ Sunday constructor
  deriving (Eq, Ord, Show, Read, Bounded, Enum)    

data Tree a
  = EmptyTree  -- ^ EmptyTree constructor
  | Node a (Tree a) (Tree a)  -- ^ internal constructor
  deriving (Show, Read, Eq)

{-| YesNo class -}
class YesNo a where
  -- | Class method taking instance of type a and returning boolean value
  yesno :: a -- ^ argument of 'a' type
        -> Bool -- ^ return value of boolean type

instance YesNo Int where
  yesno 0 = False
  yesno _ = True  


data List a = Nil | Cons a (List a)

--"a type a is an instance of the class Eq if there is an (overloaded) operation ==, of the appropriate type, defined on it." i.e., "Thus Eq a is not a type expression, but rather it expresses a constraint on a type, and is called a context."
--class Eq a where 
--  (==)  ::  a -> a -> Bool
--instance Eq Integer where
--  x == y = x `integerEq` y

-- elem:: (Eq a) -> a -> [a] -> Bool
-- read if a is instance of Eq then elem on type a has
-- that type ?expresed in type modulation?


-- it is like template class in c++, i.e., NOTE EXACTLY operator==

-- template<class a> class Mycomp { 
--   virtual bool comp(a, a); 
--   bool operator==()
-- }

class Mycomp a where
  comp :: a -> a -> Bool
--  (==) :: a -> a -> Bool

--class (Eq m, Ord m) => Set m a b where
--  exists::a->(b->Bool)->Bool
  
--class Set a b | a -> b where 
--  exists :: a -> (b -> Bool) -> Bool

class Sets f where 
  exists :: f a -> (a -> Bool) -> Bool

class Some a where
  isBigger :: Ord a => a -> a -> Bool

instance Some Integer where
  isBigger a b = a > b
-- template<> class Mycomp<int> { bool comp(int x, int y) { return x>y; }; }

instance Mycomp Integer where
  x `comp` y = x > y

instance Mycomp Char where
  x `comp` y = x > y

--instance (MyComparison a) => MyComparison (List a) where
--  minimum a == maximum a = True
--  _ == _ = False

--instance Mycomp SomeEntry where
--  x `comp` y = x > y

--instance (MyComparison a) => MyComparison (List Integer) where
--  x `comp` y = x > y

-- where s := smaller, t := than, b := bigger, e := equal
class (Mycomp a)  => (MycompChild a) where
  st0, ste, bt, bte :: (Ord a, Eq a) => a -> a -> Bool
  cmin, cmax:: Ord a => a -> a -> a
