{-|
Module: Xinh
Copyright: (c) Cezary Stankiewicz 2016  
Description: Exercises in Haskell
License: -
Maintainer: -
Stability   : -

This module contains only custom interpretations.

Use GHCi with @:load Xinh@.
To reload use @:reload@.

See $todos$.
-}
module Xinh (reverse, prepend) where
import Prelude hiding (reverse)
import Data.Char
import Data.List
import Data.List.Split
import Data.Text

reverse :: [a] -> [a]
reverse = foldl (\n -> (\x -> (x : n))) []

{-| prepend second element of tuple argument to first list tuple argument -}
prepend :: ([a], a) -> [a]
prepend = uncurry (\n -> (\x -> (x : n)))

fullWords :: Integer -> String
fullWords n = concat $ intersperse "-" [digits!!digitToInt d | d <- show n]
  where digits = ["zero", "one", "two", "three", "four",
                  "five", "six", "seven", "eight", "nine"]

-- This solution does a simple table lookup after converting the positive integer into a string. Thus dividing into digits is much simplified. 
firstTaskSplit arg = splitOn "-" arg

-- introduce coeeficient where length of splitted array signalizes power of ten
-- i.e., elem[k] * 10 ^ (size_of(elem) - k)
fullWordsInverse :: String -> Integer
fullWordsInverse n = [digits!!digitToInt d | d <- show n]
  where digits = ['0','1','2','3','4','5','6','7','8','9']

mysq :: (Num a) => [a] -> [a]
mysq [] = []
mysq (x:xs) = x*x : mysq(xs)

data Sec = Sec { geta :: Int, getb :: Int, getc :: Int } deriving (Show)

my2li xs = split (==' ') $ pack xs

myReadLi [] = 0
myReadLi (x:xs) = read x 

type Lisec = [Sec]

data Lab = A | B | C deriving (Show, Eq)
type Path = [(Lab, Int)]

-- λ. takeSt [(A,1),(B,2)] = [A,B]
takeSt :: [(a,b)] -> [a]
takeSt xs = Data.List.map fst xs

takeNd :: [(a,b)] -> [b]
takeNd xs = Data.List.map snd xs

myFindLab :: Lab -> Path -> Maybe Lab
myFindLab lab xs = Data.List.find (==lab) (takeSt xs)

myFindInt :: Int -> Path -> Maybe Int
myFindInt int xs = Data.List.find (==int) (takeNd xs)

subsets _      0 = [[]]
subsets []     _ = []
subsets (x:xs) n = map (x :) (subsets xs (n - 1)) ++ (subsets xs n)
