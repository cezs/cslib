{-|
Module: AlgebraicDataTypes
Description: (5 items)
License: To be free
Maintainer  : c.stankiewicz@wlv.ac.uk
-}
module AlgebraicDataTypes (List, Mycomp, Sets, MycompChild) where

import Data.List as List

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
  
class (Mycomp a)  => (MycompChild a) where
  st, ste, bt, bte :: (Ord a, Eq a) => a -> a -> Bool
  cmin, cmax:: Ord a => a -> a -> a


--class MYSchema a where 
--  s
