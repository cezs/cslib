## GHCi Commands

Use the following steps to load, use and retrive information about the `CS` module.

1. Load package

    ```
    ->  cphss git:(master) x ghci
    GHCi, version 7.8.4: http://www.haskell.org/ghc/  :? for help
    Loading package ghc-prim ... linking ... done.
    Loading package integer-gmp ... linking ... done.
    Loading package base ... linking ... done.
    Prelude> :l CS
    [1 of 4] Compiling AlgebraicDataTypes ( AlgebraicDataTypes.hs, interpreted )
    [2 of 4] Compiling NoviCollection   ( NoviCollection.hs, interpreted )
    [3 of 4] Compiling BasicsCollection ( BasicsCollection.hs, interpreted )
    [4 of 4] Compiling CS               ( CS.hs, interpreted )
    Ok, modules loaded: CS, BasicsCollection, NoviCollection, AlgebraicDataTypes.
    ```

2. Browse package

    ```
    *CS> :browse!
    ```

    should return, 

    ```
    -- imported via AlgebraicDataTypes
    data List a
      = AlgebraicDataTypes.Nil | AlgebraicDataTypes.Cons a (List a)
    class Mycomp a where
      AlgebraicDataTypes.comp :: a -> a -> Bool
    class Mycomp a => MycompChild a where
      AlgebraicDataTypes.st :: (Ord a, Eq a) => a -> a -> Bool
      AlgebraicDataTypes.bte :: (Ord a, Eq a) => a -> a -> Bool
      AlgebraicDataTypes.bt :: (Ord a, Eq a) => a -> a -> Bool
      AlgebraicDataTypes.ste :: (Ord a, Eq a) => a -> a -> Bool
      AlgebraicDataTypes.cmin :: Ord a => a -> a -> a
      AlgebraicDataTypes.cmax :: Ord a => a -> a -> a
    class Sets (f :: * -> *) where
      AlgebraicDataTypes.exists :: f a -> (a -> Bool) -> Bool
    -- imported via BasicsCollection
    data Car
      = BasicsCollection.Car {BasicsCollection.company :: String,
      BasicsCollection.model :: String,
      BasicsCollection.year :: Int}
    data Day
      = BasicsCollection.Monday
      | BasicsCollection.Tuesday
      | BasicsCollection.Wednesday
      | BasicsCollection.Thursday
      | BasicsCollection.Friday
      | BasicsCollection.Saturday
      | BasicsCollection.Sunday
    data Tree a
      = BasicsCollection.EmptyTree
      | BasicsCollection.Node a (Tree a) (Tree a)
    class YesNo a where
      BasicsCollection.yesno :: a -> Bool
      calcBmis :: RealFloat a => [(a, a)] -> [a]
      calcBmis2 :: RealFloat a => [(a, a)] -> [a]
      cylinder :: RealFloat a => a -> a -> a
      factorial :: Integral a => a -> a
      myConc :: [[Char]] -> [[Char]] -> [[Char]]
      myMax :: (Ord a, Num a) => a -> a -> a -> a
      quicksort :: Ord a => [a] -> [a]
      rightTriangles :: [(Integer, Integer, Integer)]
    -- imported via NoviCollection
    class Cp a where
      (NoviCollection.==) :: a -> a -> Bool
      (NoviCollection./=) :: a -> a -> Bool
    data NCList a
      = NoviCollection.Nil | NoviCollection.Cons a (NCList a)
    data NCTree a
      = NoviCollection.Node a (NCTree a) (NCTree a)
      | NoviCollection.Empty
    data Shp
      = NoviCollection.Shp1 Float Float Float
      | NoviCollection.Shp2 Float Float Float Float
    data SomeStru
      = NoviCollection.SomeStru {NoviCollection.entr1 :: String,
    NoviCollection.entr2 :: Int}
      addVec :: (Num a, Num t) => (a, t) -> (a, t) -> (a, t)
      areasq :: Num a => a -> a
      areat :: Floating a => a -> a -> a -> a
      asum :: [Integer] -> [Integer]
      classify :: (Num a, Eq a) => a -> [Char]
      doit :: Int -> Int
      doubleMe :: Num a => a -> a
      doubleSmallNumber :: (Ord a, Num a) => a -> a
      dsn :: (Ord a, Num a) => a -> a
      f :: Integer -> Integer
      fac :: (Num a, Eq a) => a -> a
      fact :: Integer -> Integer
      farrayRemap :: (Num b, Enum b) => b -> b -> b -> [b]
      farrayRev :: Enum a => a -> a -> [a]
      foo :: Floating a => a -> a
      foo2 :: Floating a => a -> a
      fsin :: Floating a => a -> a -> a
      g :: Num a => (t -> a) -> t -> a
      g2 :: Num a => (t -> a) -> (t -> a) -> t -> a
      gDown :: Integer -> [Integer]
      infinity :: a -> [a]
      isBigger :: Ord a => a -> a -> Bool
      lLen :: Num a => [t] -> a
      laugh :: [Char]
      lc :: (Ord t, Num t) => [(t, t)] -> [t]
      listlength :: [Integer] -> Integer
      lowest :: Int -> Int -> Int
      maksimum :: (Ord a, Num a) => [a] -> a
      maxList :: [Int] -> Int
      maxed :: Int -> Int -> Int -> Int -> Int
      maxi :: Int -> Int -> Int
      multThree :: Num a => a -> a -> a -> a
      multWithNine :: Integer -> Integer -> Integer
      multiplyList :: Integer -> [Integer] -> [Integer]
      myFoo :: Num a => a -> a
      myfun :: [t] -> Maybe [t]
      mylookup :: Cp a => a -> [(a, t)] -> Maybe a
      nd :: (a, b, c) -> b
      numofs :: (Ord a, Num a) => a -> a -> [Char]
      pows :: (Ord a, Num a) => a -> a -> a
      qroots :: (Ord t, Floating t) => (t, t, t) -> (t, t)
      rd :: (a, b, c) -> c
      recu :: Num t => [t] -> [t]
      recuu :: (Ord t, Num t) => t -> [t] -> [t]
      recuuu :: (Ord a, Num a) => a -> [t] -> [t]
      sq :: Num a => a -> a
      st :: (a, b, c) -> a
      surf :: Shp -> Float
      takeElems :: (Ord a, Num a) => a -> [t] -> [t]
      thatis :: (Num t, Floating [t], Enum t) => [t] -> [t]
      trig :: Floating a => a -> a
      volbox :: Num a => a -> a -> a -> a
      volcyl :: Floating a => a -> a -> a
    -- imported via NoviCollection, Prelude
      take :: Int -> [a] -> [a]
    ```
3. Use known name to retrieve information about the item

    ```
    *CS> :info quicksort
    ```

    returns, 

    ```
    quicksort :: Ord a => [a] -> [a]
    -- Defined at BasicsCollection.hs:88:1
    ```
