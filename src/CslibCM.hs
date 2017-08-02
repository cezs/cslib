{-# OPTIONS_HADDOCK show-extensions #-}
{-|
Module: CslibCM
Copyright: (c) Cezary Stankiewicz 2016
Description: Using Control.Monad module
License: -
Maintainer  : -
Stability   :  experimental

This module provides examples of use of Control.Monad module from different point of application. Currently, module provides only IO aspect of use.

Examples are to be provided for:
Control.Monad.<=<           Control.Monad.forM          Control.Monad.mplus
Control.Monad.=<<           Control.Monad.forM_         Control.Monad.msum
Control.Monad.>=>           Control.Monad.forever       Control.Monad.mzero
Control.Monad.>>            Control.Monad.guard         Control.Monad.replicateM
Control.Monad.>>=           Control.Monad.join          Control.Monad.replicateM_
Control.Monad.Functor       Control.Monad.liftM         Control.Monad.return
Control.Monad.Monad         Control.Monad.liftM2        Control.Monad.sequence
Control.Monad.MonadPlus     Control.Monad.liftM3        Control.Monad.sequence_
Control.Monad.ap            Control.Monad.liftM4        Control.Monad.unless
Control.Monad.fail          Control.Monad.liftM5        Control.Monad.void
Control.Monad.filterM       Control.Monad.mapAndUnzipM  Control.Monad.when
Control.Monad.fmap          Control.Monad.mapM          Control.Monad.zipWithM
Control.Monad.foldM         Control.Monad.mapM_         Control.Monad.zipWithM_
Control.Monad.foldM_        Control.Monad.mfilter

To find out information about all the provided definitions use the following command in ghci:
  λ> :browse! UseOfCM

which lists all the module's definitions.

NOTES:
-- https://wiki.haskell.org/Introduction_to_IO
-- https://en.wikibooks.org/wiki/Haskell/Understanding_monads
-- https://en.wikibooks.org/wiki/Haskell/Monad_transformers
-}
module CslibCM where

import Data.Char (toUpper)
import Data.Maybe
import Control.Monad
import System.Posix.Directory
import System.Posix.Files

import CslibBasics

{-|-}
main = addEnd

{-| identity -}
mainI :: IO ()
mainI = getLine >>= putStrLn

{-| convert to uppercase action ver.A0 -}
mainAver0 :: IO ()
mainAver0 = putStrLn "Write your string: " >> fmap (\x -> map toUpper x) getLine >>= putStrLn

{-| convert to uppercase action ver.B0 -}
{-| introduces dependencies: shout -}
mainBver0 :: IO ()
mainBver0 = putStrLn "Write your string: " >> fmap shout getLine >>= putStrLn

{-| convert to uppercase action ver.1 -}
mainA :: IO ()
mainA = fmap (\x -> map toUpper x) getLine >>= putStrLn

{-| convert to uppercase action ver.2 -}
{-| introduces dependencies: shout -}
mainB :: IO ()
mainB = fmap shout getLine >>= putStrLn

{-| convert to uppercase action ver.3 -}
{-| introduces dependencies: shout -}
mainC :: IO ()
mainC = liftM shout getLine >>= putStrLn

{-| convert to uppercase action ver.4 -}
{-| introduces dependencies: shout, works: on whole content -}
mainD :: IO ()
mainD = liftM shout getContents >>= putStr

{-| convert to uppercase action ver.5 -}
mainE :: IO ()
mainE = fmap (map toUpper) getLine >>= putStrLn

{-| convert to uppercase -}
shout :: [Char] -> [Char]
shout = map toUpper

{-| Stub 1 -}
moder :: [Char] -> [Char]
moder = Prelude.reverse

{-| if compiled, try @man man | ./dmnt | ./dmnt@ -}
uglifier :: IO ()
uglifier = liftM (moder . shout) getContents >>= putStrLn

{-| turn into tuple of @Int@ and @String@ -}
{-| example use: -}
{-| sumaSeq = forM_ ["1aa","2cc","3b"] (liftM sumator) -}
sumator :: String -> [(Int, String)]
sumator v = reads v :: [(Int, String)]

curio :: Read b => IO b
curio = liftM read getContents >>= return

{-| Perhaps use @csactions [putStr "Hello ", putStr "World.\n"]@
or @csactions [Nothing, Just $ Nothing]@ -}
csactions :: Monad m => [m a] -> m ()
csactions = foldr (>>) (return ())

{-|-}
csActionStack :: Monad m => m () -> [m ()]
csActionStack a = a : [return ()]

{-|-}
csActionStack5 :: Monad m => [m ()] -> [m ()]
csActionStack5 a = a ++ [return ()]

{-|-}
-- csActionStack4 :: Monad m => m () -> [m ()]
-- csActionStack4 xs = [y | x <- xs, y <- [return()]]

{-|-}
csActionStack2 :: a -> [a] -> [a]
csActionStack2 a lsa@(x:xs) = a : lsa


{-|-}
csActionStack3 :: Monad m => m () -> [m ()]
csActionStack3 a = csActionStack2 a [return ()]

{-| Stub 2 -}
-- instance Monad Tree where
-- instance Monad Tree

{-|-}
addEnd = getLine >>= return . (\str -> str ++ "\n") >>= putStr

{-|-}
-- given monadic value either:
-- -- allow a function to operating on it through #liftM
-- -- or use #bind operator to pass it to a function having non monadic arguments
-- ---- but then you have to monadize its output using #return if it is not already monadic
-- see: addEnd2, addEnd3
addEnd2 = getLine >>= return . (\str -> str ++ "\n") >>= putStr

{-|-}
addEnd3 = liftM (\str -> str ++ "\n") getLine >>= putStr

{-|-}
addEnd4 = (return . (\str -> str ++ "\n") =<< getLine) >>= putStr

{-|-}
addEnd5 = getLine >>= return . (++"\n") >>= putStr

{-| shortest so far -}
addEnd6 = liftM (++"\n") getLine >>= putStr

-- liftM
--   (nonmonadic function, monadic value) -> monadic value
-- return
--   nonmonadic value -> monadic value

{-| convert to uppercase action ver.6 -}
{-| introducing 'do' notation, with procedural-like -}
{-| style and stricter requirements on indentation -}
{-| given we are not using either bractes {} or semicolon ; -}
myMain :: IO ()
myMain = do
  x <- getLine
  s <- return $ shout x
  putStrLn s

-- example of recursive action from 'Beautiful Concurrecny' p.7
-- performs 'action' action n times
-- example:
--     λ> nTimes 3 $ putStrLn "Hey"
-- "In effect, by treating actions as first-class values, Haskell
-- "supports user-defined control structures."
nTimes :: Int -> IO () -> IO ()
nTimes 0 action = return ()
nTimes n action = do { action; nTimes (n-1) action }

{-| Stub 1 -}
newtype MyConf = MyConf { vao :: Bool } deriving (Show)

{-| Defaults -}
myConfDef :: MyConf
myConfDef = MyConf { vao = False }

{-| Homogeneous -}
myConfLs :: MyConf -> MyConf
myConfLs ls@(MyConf { vao = v }) = ls

{-| Monadic 1 -}
{-| Action 1 -}
act :: Monad m => m MyConf
act = liftM return myConfLs $ myConfDef
{-| where myConfLs $ myConfDef :: MyConf -}
{-| Action 2 -}
act2 :: Monad m => m MyConf
act2 = liftM myConfLs (return $ myConfDef)

{-|-}
data TreeF a = NodeF { rootName :: a, hForestF :: ForestF a }

{-|-}
type ForestF a = [TreeF a]

{-|-}
instance Functor TreeF where
  fmap = fmapTreeF

{-|-}
fmapTreeF :: (t -> a) -> TreeF t -> TreeF a
fmapTreeF f (NodeF x xs) = NodeF (f x) (map (fmapTreeF f) xs)

mainHio = return factorial 2

fooHio v
  | v <= 0 = Just v
  | otherwise = Nothing

barHio :: (Ord a, Num a) => a -> Maybe a

barHio v
  | v <= 0 = Nothing
  | otherwise = Just v

ttHio n = do
  k <- fooHio n
  b <- barHio k
  return b

t2Hio n = fooHio n >>= barHio

data Conf =  Conf
             { cid :: Int
             , name :: String
             }

pairOff :: Int -> Either String Int
pairOff people
    | people < 0  = Left "Can't pair off negative number of people."
    | people > 30 = Left "Too many people for this activity."
    | even people = Right (people `div` 2)
    | otherwise   = Left "Can't pair off an odd number of people."

groupPeople :: Int -> String
groupPeople people = case pairOff people of
                       Right groups -> "We have " ++ show groups ++ " group(s)."
                       Left problem -> "Problem! " ++ problem

nameReturn :: IO String
nameReturn = do
  let name = "Hwsws.W"
  return name

see :: String -> String
see s = s ++ "ioskwok"

ss :: IO ()
ss = fmap see nameReturn >>= putStrLn

-- sess :: IO()
-- sess = do
--   putStrLn "Enter string to be changed to uppercase"
--   name <- getLine
--   if null name
--      then return ()
--   else do
--      let bigName = map toUpper name
--      putStrLn ("You have just entered" ++ " " ++ bigName)
--      sess
