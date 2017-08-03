{-# LANGUAGE MultiParamTypeClasses, TypeFamilies #-}
{-|
Module: CslibBasics
Description: -
License: -
Maintainer: c.stankiewicz@wlv.ac.uk

Introductory examples

Use GHCi with @:load BasicsCollection@.
To reload use @:reload@.

Also, see $todos$.
-}
module CslibBasics where

import Prelude
import Data.Char
import Data.Set
import Data.IntSet
import Data.List
import Data.List.Split
import Data.Text


-- functions -------------------------------------------------------------------

{-|-}
laugh :: [Char]
laugh = "Ha!"

{-|-}
sq :: Num a => a -> a
sq x = x * x

{-|-}
vol :: Num a => a -> a -> a -> a
vol x y z  = x * y * z

{-|-}
height :: Floating a => a -> a -> a
height a r = sin(a) * r

{-|-}
peach x = [1..10]**x

{-|-}
addthree :: Integer -> Integer
addthree = (+3)

{-|-}
g :: Num a => (t -> a) -> t -> a
g f x = f x * f x

{-|-}
g2 :: Num a => (t -> a) -> (t -> a) -> t -> a
g2 f1 f2 x = f1 x + f2 x

{-|-}
multWithTen :: Integer -> Integer -> Integer
multWithTen = vol 10

{-|-}
revlist :: Enum a => a -> a -> [a]
revlist a z = Data.List.reverse [a .. z]

{-|-}
fact :: Integer -> Integer
fact n = product [1..n]

{-|-}
addVec :: (Num a, Num t) => (a, t) -> (a, t) -> (a, t)
addVec a b = (fst a + fst b, snd a + snd b)

-- conditional -----------------------------------------------------------------

{-|-}
pows :: (Num a, Ord a) => a -> a -> a
pows n m = if m > 0 then m - 1 else n * n

{-|-}
dsn :: (Num a, Ord a) => a -> a
dsn x = (if x > 100 then x else x * 2) + 1

-- where -----------------------------------------------------------------------

{-|-}
foo n = n * pi * s / c
    where s = n * 2
          c = n * 4

{-|-}
volofcyl r y = areac * y  where areac = sq (pi  * r)

{-|-}
qroots (a, b, c) =
  if d < 0 then error "confused"
    else (r1, r2)
      where r1 = e + sqrt d / (2 * a)
            r2 = e - sqrt d / (2 * a)
            d = b * b - 4 * a * c
            e = -b / (2 * a)

{-|-}
numofs a b
    | num > 0 = "pos"
    | num < 0 = "neg"
    | otherwise = "neu"
        where
        num = a + b

-- let-in ----------------------------------------------------------------------

{-|-}
foo2 n = let a = 4 / 5
             cube = n ^ 3
         in a * pi * cube

{-|-}
cylinder :: (RealFloat a) => a -> a -> a  
cylinder r h = 
  let sideArea = 2 * pi * r * h
      topArea = pi * r^2
  in  sideArea + 2 * topArea  

-- case-of ---------------------------------------------------------------------

{-|-}
classify :: (Num a, Eq a) => a -> [Char]
classify age = case age of 0 -> "newborn"
                           1 -> "infant"
                           2 -> "toddler"
                           _ -> "..."

-- guards -----------------------------------------------------------------------

{-|-}
maxi :: Int -> Int -> Int
maxi a b | a > b = a
         | otherwise = b

{-|-}
maxList ::  [Int] -> Int
maxList []  = 0
maxList (a:az)
   | maxList az > a = maxList az
   | otherwise = a

{-|-}
multBy _ [] = []
multBy n _
    | n <= 0 = []
multBy n (x:xs) = n * x : multBy n xs

{-|-}
getSubset _ [] = []
getSubset n _
    | n <= 0 = []
getSubset n (x:xs) = x : getSubset (n-1) xs

{-|-}
factorial :: (Integral a) => a -> a
factorial n | n < 2 = 1
factorial n = n * factorial (n - 1)

{-|-}
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
sqEach [] = []
sqEach (x:xs) = x * x : sqEach xs

{-|-}
nonstop :: a -> [a]
nonstop x = x : nonstop x

{-|-}
take' x n = getSubset x (infinity n)

{-|-}
maksimum :: (Ord a, Num a) => [a] -> a
maksimum [] = 0
maksimum (current_maksimum : container_has)
         | maksimum container_has > current_maksimum
           = maksimum container_has
             | otherwise = current_maksimum

{-|-}
gDown :: Integer -> [Integer]
gDown (-5) = [-5]
gDown n = n : gDown (n - 1)

{-|-}
myDrop :: Int -> [a] -> [a]
myDrop n xs = if n <= 0 || Prelude.null xs then xs else myDrop (n - 1) (Data.List.tail xs)

{-|-}
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
fac :: (Num a, Eq a) => a -> a
fac 0 = 1
fac n = n * fac(n-1)

-- list comprehesion -----------------------------------------------------------

lc xs = [foo | (n, m) <- xs, let foo = n + m, foo >= 10]

rightTriangles = [(a,b,c) | c <- [1..128], b <- [1..c], a <- [1..b], a^2 + b^2 Prelude.== c^2 ]

myConc przym rzecz = [prz ++ " " ++ rz | prz <- przym, rz <- rzecz]

calcBmis :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis xs = [bmi w h | (w, h) <- xs]
  where bmi weight height = weight / height ^ 2  

calcBmis2 :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis2 xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2, bmi >= 25.0]  

-- functorial -----------------------------------------------------------------
          
{-|-}
farrayRemap mov f t = Data.List.map (+mov) [f .. t]

-- monadic values --------------------------------------------------------------

{-|-}
myfun [] = Nothing
myfun a = Just a

{-|-}
mylookup _ [] = Nothing
mylookup key ((x,y):s) = if key Prelude.== x then Just x else mylookup key s

-- data, type, class, instance -------------------------------------------------

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
    x == y = not ( x CslibBasics./= y)

type Endomorphic a = a -> a

-- itsme :: (Endomorphic a) => a -> a
-- itsme a = a

{-|-}
class Equal a where
  eq, neq :: a -> a -> Bool

{-|-}
instance Equal Bool where
  true `eq` b = b
  false `eq` b = not b
  c `neq` b = not (c `eq` b)

{-|-}
class Collection c where
  type Element c :: *
  cmember :: Element c -> c-> Bool

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

-- other -----------------------------------------------------------------------

reverse :: [a] -> [a]
reverse = Data.List.foldl (\n -> (\x -> (x : n))) []

{-| prepend second element of tuple argument to first list tuple argument -}
prepend :: ([a], a) -> [a]
prepend = uncurry (\n -> (\x -> (x : n)))

fullWords :: Integer -> String
fullWords n = Data.List.concat $ Data.List.intersperse "-" [digits!!digitToInt d | d <- show n]
  where digits = ["zero", "one", "two", "three", "four",
                  "five", "six", "seven", "eight", "nine"]

-- firstTaskSplit arg = Data.Text.splitOn "-" arg

-- fullWordsInverse :: String -> Integer
-- fullWordsInverse n = [digits!!digitToInt d | d <- show n]
--   where digits = ['0','1','2','3','4','5','6','7','8','9']

mysq :: (Num a) => [a] -> [a]
mysq [] = []
mysq (x:xs) = x*x : mysq(xs)

data Sec = Sec { geta :: Int, getb :: Int, getc :: Int } deriving (Show)

my2li xs = Data.Text.split (Prelude.==' ') $ pack xs

myReadLi [] = 0
myReadLi (x:xs) = read x 

type Lisec = [Sec]

data Lab = A | B | C deriving (Show, Eq)
type Path = [(Lab, Int)]

takeSt :: [(a,b)] -> [a]
takeSt xs = Data.List.map fst xs

takeNd :: [(a,b)] -> [b]
takeNd xs = Data.List.map snd xs

myFindLab :: Lab -> Path -> Maybe Lab
myFindLab lab xs = Data.List.find (Prelude.==lab) (takeSt xs)

myFindInt :: Int -> Path -> Maybe Int
myFindInt int xs = Data.List.find (Prelude.==int) (takeNd xs)

subsets _      0 = [[]]
subsets []     _ = []
subsets (x:xs) n = Data.List.map (x :) (subsets xs (n - 1)) ++ (subsets xs n)
