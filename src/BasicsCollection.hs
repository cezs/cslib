{-|
Module: BasicsCollection
Description: First module written in Haskell language (18 items)
License: To be free
Maintainer  : c.stankiewicz@wlv.ac.uk

This module defines types 'Day', 'Tree', a class 'YesNo', a record 'Car'.

Use GHCi with @:load BasicsCollection@.

To reload use @:reload@.

Also, see $todos$.
-}
module BasicsCollection (
  -- * Functions
  main,
  factorial,
  rightTriangles,
  myConc,
  myMax,
  calcBmis,
  cylinder,
  calcBmis2,
  quicksort,
  -- * A record
  Car,
  -- * Types
  Day,
  Tree,
  -- * A class
  YesNo
) where

import Data.Char

-- TODO: Move to new module ProceduralSessions
main :: IO()
main = do
  putStrLn "Enter string to be changed to uppercase"
  name <- getLine
  if null name
     then return ()
  else do
     let bigName = map toUpper name
     putStrLn ("You have just entered"
               ++ " "
               ++ bigName
               )
            -- ++ "asking for uppercase version"
     main

factorial :: (Integral a) => a -> a
factorial n | n < 2 = 1
factorial n = n * factorial (n - 1)

rightTriangles = [(a,b,c) | c <- [1..128], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2 ]

myConc przym rzecz = [prz ++ " " ++ rz | prz <- przym, rz <- rzecz]

myMax a b c
  | c > d = c
  | otherwise = d
  where d = a * b

calcBmis :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis xs = [bmi w h | (w, h) <- xs]
  where bmi weight height = weight / height ^ 2  

cylinder :: (RealFloat a) => a -> a -> a  
cylinder r h = 
  let sideArea = 2 * pi * r * h
      topArea = pi * r ^2
  in  sideArea + 2 * topArea  

calcBmis2 :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis2 xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2, bmi >= 25.0]  


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

-- lambda
-- map (\(a,b) -> a + b) [(1,2),(3,5),(6,3),(2,6),(2,5)]

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

{-|
$todos
TODO
-- module A (
--   module B,
--   module C
--  ) where

-- import B hiding (f)
-- import C (a, b)
$todos
-}
