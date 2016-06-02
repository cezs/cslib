-- lambda calculus derived languages
-- lisp, haskell
-- in application called functional
-- nothing like a procedural thing, here we show computer how to calculate something using patterns


-- ok, lets break it down
-- Given the following --
-- λ> :i Monad
-- class Monad (m :: * -> *) where
--   (>>=) :: m a -> (a -> m b) -> m b
--   (>>) :: m a -> m b -> m b
--   return :: a -> m a
--   fail :: String -> m a
--   	-- Defined in ‘GHC.Base’
-- instance Monad Maybe -- Defined in ‘Data.Maybe’
-- instance Monad (Either e) -- Defined in ‘Data.Either’
-- instance Monad [] -- Defined in ‘GHC.Base’
-- instance Monad IO -- Defined in ‘GHC.Base’
-- instance Monad ((->) r) -- Defined in ‘GHC.Base’


-- The one that reads

--   (>>=) :: m a -> (a -> m b) -> m b

-- could be read as follows, apply function `m` to `a` where `a` is a function of `m` applied to `b` and return in fact `m` applied to `b`.

module Shapes ( Point(..), Shape(..), surface, shift ) where

-- 'Circle', 'Other' := Value Constructors, return 'Shape'
-- i.e., Circle :: Point -> Float -> Shape
data Point = Point Float Float deriving (Show)
data Shape = Circle Point Float | Other Point Point deriving (Show)

-- to run call either, surface(Circle (Point 1 1) 1) or surface $ Circle (Point 1 1) 1
surface :: Shape -> Float
surface (Circle _ r) = pi * r * r

shift (Circle (Point x y) r) a b = Circle (Point (x+a) (y+b)) r


-- Click on a completion to select it.
-- In this buffer, type RET to select the completion near point.

-- Possible completions are:
-- !! 	$ 	$!
-- && 	* 	**
-- + 	++ 	-
-- . 	/ 	/=
-- < 	<= 	=<<
-- == 	> 	>=
-- >> 	>>= 	Bool
-- Bounded 	Char 	Circle
-- Double 	EQ 	Either
-- Enum 	Eq 	False
-- FilePath 	Float 	Floating
-- Fractional 	Functor 	GT
-- IO 	IOError 	Int
-- Integer 	Integral 	Just
-- LT 	Left 	Maybe
-- Monad 	Nothing 	Num
-- Ord 	Ordering 	Prelude.!!
-- Prelude.$ 	Prelude.$! 	Prelude.&&
-- Prelude.* 	Prelude.** 	Prelude.+
-- Prelude.++ 	Prelude.- 	Prelude..
-- Prelude./ 	Prelude./= 	Prelude.<
-- Prelude.<= 	Prelude.=<< 	Prelude.==
-- Prelude.> 	Prelude.>= 	Prelude.>>
-- Prelude.>>= 	Prelude.Bool 	Prelude.Bounded
-- Prelude.Char 	Prelude.Double 	Prelude.EQ
-- Prelude.Either 	Prelude.Enum 	Prelude.Eq
-- Prelude.False 	Prelude.FilePath 	Prelude.Float
-- Prelude.Floating 	Prelude.Fractional 	Prelude.Functor
-- Prelude.GT 	Prelude.IO 	Prelude.IOError
-- Prelude.Int 	Prelude.Integer 	Prelude.Integral
-- Prelude.Just 	Prelude.LT 	Prelude.Left
-- Prelude.Maybe 	Prelude.Monad 	Prelude.Nothing
-- Prelude.Num 	Prelude.Ord 	Prelude.Ordering
-- Prelude.Rational 	Prelude.Read 	Prelude.ReadS
-- Prelude.Real 	Prelude.RealFloat 	Prelude.RealFrac
-- Prelude.Right 	Prelude.Show 	Prelude.ShowS
-- Prelude.String 	Prelude.True 	Prelude.^
-- Prelude.^^ 	Prelude.abs 	Prelude.acos
-- Prelude.acosh 	Prelude.all 	Prelude.and
-- Prelude.any 	Prelude.appendFile 	Prelude.asTypeOf
-- Prelude.asin 	Prelude.asinh 	Prelude.atan
-- Prelude.atan2 	Prelude.atanh 	Prelude.break
-- Prelude.ceiling 	Prelude.compare 	Prelude.concat
-- Prelude.concatMap 	Prelude.const 	Prelude.cos
-- Prelude.cosh 	Prelude.curry 	Prelude.cycle
-- Prelude.decodeFloat 	Prelude.div 	Prelude.divMod
-- Prelude.drop 	Prelude.dropWhile 	Prelude.either
-- Prelude.elem 	Prelude.encodeFloat 	Prelude.enumFrom
-- Prelude.enumFromThen 	Prelude.enumFromThenTo 	Prelude.enumFromTo
-- Prelude.error 	Prelude.even 	Prelude.exp
-- Prelude.exponent 	Prelude.fail 	Prelude.filter
-- Prelude.flip 	Prelude.floatDigits 	Prelude.floatRadix
-- Prelude.floatRange 	Prelude.floor 	Prelude.fmap
-- Prelude.foldl 	Prelude.foldl1 	Prelude.foldr
-- Prelude.foldr1 	Prelude.fromEnum 	Prelude.fromInteger
-- Prelude.fromIntegral 	Prelude.fromRational 	Prelude.fst
-- Prelude.gcd 	Prelude.getChar 	Prelude.getContents
-- Prelude.getLine 	Prelude.head 	Prelude.id
-- Prelude.init 	Prelude.interact 	Prelude.ioError
-- Prelude.isDenormalized 	Prelude.isIEEE 	Prelude.isInfinite
-- Prelude.isNaN 	Prelude.isNegativeZero 	Prelude.iterate
-- Prelude.last 	Prelude.lcm 	Prelude.length
-- Prelude.lex 	Prelude.lines 	Prelude.log
-- Prelude.logBase 	Prelude.lookup 	Prelude.map
-- Prelude.mapM 	Prelude.mapM_ 	Prelude.max
-- Prelude.maxBound 	Prelude.maximum 	Prelude.maybe
-- Prelude.min 	Prelude.minBound 	Prelude.minimum
-- Prelude.mod 	Prelude.negate 	Prelude.not
-- Prelude.notElem 	Prelude.null 	Prelude.odd
-- Prelude.or 	Prelude.otherwise 	Prelude.pi
-- Prelude.pred 	Prelude.print 	Prelude.product
-- Prelude.properFraction 	Prelude.putChar 	Prelude.putStr
-- Prelude.putStrLn 	Prelude.quot 	Prelude.quotRem
-- Prelude.read 	Prelude.readFile 	Prelude.readIO
-- Prelude.readList 	Prelude.readLn 	Prelude.readParen
-- Prelude.reads 	Prelude.readsPrec 	Prelude.realToFrac
-- Prelude.recip 	Prelude.rem 	Prelude.repeat
-- Prelude.replicate 	Prelude.return 	Prelude.reverse
-- Prelude.round 	Prelude.scaleFloat 	Prelude.scanl
-- Prelude.scanl1 	Prelude.scanr 	Prelude.scanr1
-- Prelude.seq 	Prelude.sequence 	Prelude.sequence_
-- Prelude.show 	Prelude.showChar 	Prelude.showList
-- Prelude.showParen 	Prelude.showString 	Prelude.shows
-- Prelude.showsPrec 	Prelude.significand 	Prelude.signum
-- Prelude.sin 	Prelude.sinh 	Prelude.snd
-- Prelude.span 	Prelude.splitAt 	Prelude.sqrt
-- Prelude.subtract 	Prelude.succ 	Prelude.sum
-- Prelude.tail 	Prelude.take 	Prelude.takeWhile
-- Prelude.tan 	Prelude.tanh 	Prelude.toEnum
-- Prelude.toInteger 	Prelude.toRational 	Prelude.truncate
-- Prelude.uncurry 	Prelude.undefined 	Prelude.unlines
-- Prelude.until 	Prelude.unwords 	Prelude.unzip
-- Prelude.unzip3 	Prelude.userError 	Prelude.words
-- Prelude.writeFile 	Prelude.zip 	Prelude.zip3
-- Prelude.zipWith 	Prelude.zipWith3 	Prelude.||
-- Rational 	Read 	ReadS
-- Real 	RealFloat 	RealFrac
-- Rectangle 	Right 	Shape
-- Show 	ShowS 	String
-- True 	^ 	^^
-- abs 	acos 	acosh
-- all 	and 	any
-- appendFile 	asTypeOf 	asin
-- asinh 	atan 	atan2
-- atanh 	break 	ceiling
-- compare 	concat 	concatMap
-- const 	cos 	cosh
-- curry 	cycle 	decodeFloat
-- div 	divMod 	drop
-- dropWhile 	either 	elem
-- encodeFloat 	enumFrom 	enumFromThen
-- enumFromThenTo 	enumFromTo 	error
-- even 	exp 	exponent
-- fail 	filter 	flip
-- floatDigits 	floatRadix 	floatRange
-- floor 	fmap 	foldl
-- foldl1 	foldr 	foldr1
-- fromEnum 	fromInteger 	fromIntegral
-- fromRational 	fst 	gcd
-- getChar 	getContents 	getLine
-- head 	id 	import
-- init 	interact 	ioError
-- isDenormalized 	isIEEE 	isInfinite
-- isNaN 	isNegativeZero 	it
-- iterate 	last 	lcm
-- length 	let 	lex
-- lines 	log 	logBase
-- lookup 	map 	mapM
-- mapM_ 	max 	maxBound
-- maximum 	maybe 	min
-- minBound 	minimum 	mod
-- negate 	not 	notElem
-- null 	odd 	or
-- otherwise 	pi 	pred
-- print 	product 	properFraction
-- putChar 	putStr 	putStrLn
-- quot 	quotRem 	read
-- readFile 	readIO 	readList
-- readLn 	readParen 	reads
-- readsPrec 	realToFrac 	recip
-- rem 	repeat 	replicate
-- return 	reverse 	round
-- scaleFloat 	scanl 	scanl1
-- scanr 	scanr1 	seq
-- sequence 	sequence_ 	show
-- showChar 	showList 	showParen
-- showString 	shows 	showsPrec
-- significand 	signum 	sin
-- sinh 	snd 	span
-- splitAt 	sqrt 	subtract
-- succ 	sum 	surface
-- tail 	take 	takeWhile
-- tan 	tanh 	toEnum
-- toInteger 	toRational 	truncate
-- uncurry 	undefined 	unlines
-- until 	unwords 	unzip
-- unzip3 	userError 	words
-- writeFile 	zip 	zip3
-- zipWith 	zipWith3 	||
