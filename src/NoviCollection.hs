{-|
Module: NoviCollection
Description: Initial Attempts (60 items)
License: To be free
Maintainer  : c.stankiewicz@wlv.ac.uk
-}
module NoviCollection (doit, classify, farrayRemap, farrayRev, fsin, pows, listlength, NCList, fac, f, g, g2, maxi, maxed, lowest, maxList, lc, recu, recuu, recuuu, infinity, take, multThree, multWithNine, maksimum, takeElems, gDown, lLen, asum, st, nd, rd, multiplyList, sq, volbox, volcyl, areasq, areat, numofs, Shp, surf, SomeStru, Cp, NCTree, foo, foo2, qroots, fact, addVec, myFoo, isBigger, doubleMe, doubleSmallNumber, trig, dsn, laugh, thatis, myfun, mylookup) where

import Prelude hiding ((==), (/=))

doit :: Int -> Int
doit x  = x * i
     where
     i = x^3
     
classify age = case age of 0 -> "newborn"
                           1 -> "infant"
                           2 -> "toddler"
                           _ -> "..."

-- p = filt_pinP [2..]
--   where filt_pinP (p:xs) = 
--           p : filt_pinP [x | x <- xs, x `mod` p /= 0]
          
-- -- -- parse error on input ‘if’
-- -- factoral n   if n = 0
-- -- 	     then 1
-- --              else n * factoral ( n - 1 )

-- parse error on input ‘if’
-- factoral n = (if n = 0 then 1 else n * factoral (n - 1))

farrayRemap mov f t = map(+mov) [f .. t]

farrayRev a z = reverse [a .. z]

-- -- either distill or perfuse
-- farrayDistil model f t = filter (mod model) [f .. t]

fsin a r = (sin(a) * r)   

pows n m = if m > 0
	      then m--
              else n * n

listlength :: [Integer] -> Integer
listlength [] = 0
listlength (x:xs) = 1 + listlength xs

--------------------------------------------------------------------------------

data NCList a = Nil | Cons a (NCList a)

fac::(Num a, Eq a) => a -> a
fac 0 = 1
fac n = n * fac(n-1)

f = (+3)

g function x = function x * function x

g2 :: Num a => (t -> a) -> (t -> a) -> t -> a
g2 f1 f2 x = f1 x + f2 x

--------------------------------------------------------------------------------

maxi :: Int -> Int -> Int
maxi a b | a > b = a
         | otherwise = b

maxed :: Int -> Int -> Int -> Int -> Int
maxed a b c d = maxi a (maxi b (maxi c d))

lowest :: Int -> Int -> Int
lowest a b | a < b = a
           | otherwise = b 

maxList ::  [Int] -> Int
maxList []  = 0
maxList (a:az)
   | maxList az > a = maxList az
   | otherwise = a


lc xs = [foo | (n, m) <- xs, let foo = n + m, foo >= 10]


recu [] = []
recu (x:xs) = x * x : recu xs


recuu _ [] = []
recuu n _
    | n <= 0 = []
recuu n (x:xs) = n * x : recuu n xs


recuuu _ [] = []
recuuu n _
    | n <= 0 = []
recuuu n (x:xs) = x : recuuu (n-1) xs

infinity :: a -> [a]
infinity x = x : infinity x

--takitfromloop
take' x n = recuuu x (infinity n)

multThree x y z= x * y * z
multWithNine = multThree 9

--------------------------------------------------------------------------------

maksimum [] = 0
maksimum (current_maksimum : container_has)
         | maksimum container_has > current_maksimum
           = maksimum container_has
             | otherwise = current_maksimum

takeElems _ [] = []
takeElems n _
	| n <= 0 = []
takeElems n (x:xs) = x : takeElems (n-1) xs

--------------------------------------------------------------------------------

gDown :: Integer -> [Integer]
gDown (-5) = [-5]
gDown n = n : gDown (n - 1)

lLen [] = 0
lLen (x:y:z:o:xs) =  1 + lLen xs

-- Note: xs needs at leas 4 elements 
asum :: [Integer] -> [Integer]
asum [] = []
asum (x:y:z:o:xs) = x + x : asum xs

-- Scheme: put n for every xy pair of xs
-- double(x + x) first(x) value,[from neverending loop of four(x:y:z:o)] every first in (xyzo) in list(xs)

--------------------------------------------------------------------------------

st :: (a, b, c) -> a
st (x , _, _) = x

nd :: (a, b, c) -> b
nd (_ , y, _) = y

rd :: (a, b, c) -> c
rd (_ , _, z) = z

--------------------------------------------------------------------------------

--takeInt :: Integer -> [Integer] -> [Integer]
--takeInt _ []    =    []
--takeInt n (x:xs) = x : takeInt xs

multiplyList :: Integer -> [Integer] -> [Integer]
multiplyList _ [] = []
multiplyList m (n:ns) = (multiplyByM n) : multiplyList m ns
    where
    multiplyByM x = m * x

--------------------------------------------------------------------------------

sq x = x * x

volbox x y z  = x * y * z

volcyl r y = areac * y
     where
     areac = sq (pi  * r)

areasq x = x * x

areat a b c = sqrt ( s * (s - a) * (s - b) * (s - c) )
    where
    s = (a + b + c) / 2
	
numofs a b
    | num > 0 = "pos"
    | num < 0 = "neg"
    | otherwise = "neu"
        where
        num = a + b

--------------------------------------------------------------------------------

-- Error: Illegal literal in type (use DataKinds to enable): 0
-- data Binr = 0 | 1

data Shp = Shp1 Float Float Float | Shp2 Float Float Float Float

surf :: Shp -> Float
surf (Shp1 _ _ r) = pi * r * r
surf (Shp2 c1 c2 b1 b2) = (b1 - c1) * (b2 - c2)

data SomeStru = SomeStru { entr1 :: String, entr2 :: Int } deriving (Show)

-- Possible use:
-- 
-- @
-- instance Cp Dire where
--     Ent1 == Ent1 = True
--     Ent2 == Ent2 = True
--     Ent3 == Ent3 = True
--     _ == _ = False
-- @
class Cp a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x == y = not ( x /= y)

--------------------------------------------------------------------------------

data NCTree a = Node a (NCTree a) (NCTree a)
            | Empty
              deriving (Show)

-- Problem 9:Determine the height of a tree
-- height :: NCTree -> Int
-- height smoo (Node node left right) = if (left == Empty && right == Empty) then 0 else max (height left) (height right) 

--------------------------------------------------------------------------------

foo n = n * pi * s / c
    where s = n * 2
          c = n * 4

foo2 n = let a = 4 / 5
             cube = n ^ 3
         in a * pi * cube

qroots (a, b, c) =
  if d < 0 then error "confused"
    else (r1, r2)
      where r1 = e + sqrt d / (2 * a)
            r2 = e - sqrt d / (2 * a)
            d = b * b - 4 * a * c
            e = -b / (2 * a)

--------------------------------------------------------------------------------

-- smok t d = t*(smok( d t ) * smok( t d ))*d
-- smok' t' d' k s =(t : d) : (d : t)

--------------------------------------------------------------------------------

fact :: Integer -> Integer
fact n = product [1..n]

addVec a b = (fst a + fst b, snd a + snd b)

--------------------------------------------------------------------------------

-- class ArrayElem e where
--       data Array e
--       index :: Array e -> Int -> e

-- instance ArrayElem Int where
--          data Array Int = IntArray UIntArr
--          index (IntArray ar) i = indexUIntArr ar i

-- instance (ArrayElem a, ArrayElem b) => ArrayElem (a, b) where
--          data Array (a, b) = PairArray (Array a) (Array b)
--          index (PairArray ar br) i = (index ar i, index br i)

--------------------------------------------------------------------------------

myFoo :: Num a => a -> a
myFoo n = 1 + n

isBigger :: Ord a => a -> a -> Bool
isBigger a b = a > b

doubleMe x = x + x

doubleSmallNumber x = if x > 1
                        then x
                        else doubleMe x

trig a = sin a

dsn x = (if x > 100 then x else x*2)+1

laugh = "Heheoha"

thatis x = [1..10]**x

--------------------------------------------------------------------------------

-- lista [i] i = 0
--          if' i < 100
--            if' ++i
--            if' lista [i]

--------------------------------------------------------------------------------

myfun [] = Nothing
myfun a = Just a

mylookup _ [] = Nothing
mylookup key ((x,y):s) =
  if key == x
     then Just x
          else mylookup key s

--------------------------------------------------------------------------------

-- -- -- parse error on input ‘if’
-- -- myDrop :: Int -> [a] -> [a]
-- -- myDrop n xs = a if n <= 0 || null xs
-- --                then xs
-- -- 	       else myDrop (n - 1) (tail xs)
                                                   
--------------------------------------------------------------------------------

-- fadd n m = fadd (n ^ m) * fadd(n mod m)
-- fadd n m = fadd (n m) + fadd (m n)
-- fadd n m =  (n + m) + (n + m)
