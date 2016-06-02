module MyHaskell where

{-# LANGUAGE MultiParamTypeClasses, TypeFamilies #-}
import Data.Char
--import Data.Natural
--import Data.Stream  hiding (map)
import Data.Set     hiding (map)
import Data.IntSet  hiding (map)

-- useful ghci commands :type
-- if curious use :kind to find out whether it is * kind or * -> * etc
-- ex. :k Char :k [] :k Monad :k (->)
-- also :info Monad
-- :browse -- returns keywords

type Endomorphic a = a -> a

class Equal a where
--  data TT a :: *->*
  eq, neq :: a -> a -> Bool

instance Equal Bool where
  true `eq` b = b
  false `eq` b = not b
  c `neq` b = not (c == b)

data List a = Nil | Cons a (List a)

--itsme:: (Endomorphic a) => a -> a
itsme a = a

class Collection c where
  type Element c :: *
  cmember::Element c->c->Bool
instance Ord a => Collection (Set a) where
  type Element (Set a) = a
  cmember = Data.Set.member


type Id = Int
type Name = String

data Stru = Id Name 
