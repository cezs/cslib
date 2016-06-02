module ProceduralSessions where

import system.environment
import prelude

import stylescode

-------------------------------------------------------------------------------

-- turtle1 = box (cKurve 10 2)
--                           # setBGColor white
--                           # setDX 10
--                           # setDY 10
--                           # setShadow (2,-2)
-- cKurve 0 l = forward l
-- cKurve g l = toleft & cKurve (g - 1) l
--            & toright & cKurve (g - 1) l
           
-------------------------------------------------------------------------------

factorial n = if n == 0 then 1 else n * factorial (n - 1)
 
main1 = do putStrLn "What is 5! ?"
          x <- readLn
          if x == factorial 5
              then putStrLn "You're right!"
              else putStrLn "You're wrong!"

-------------------------------------------------------------------------------

testp pf = do print $ pf [3,7..][1..10]

test takelist n = takelist (take n[1..])[1..]


main2 = do
  (n:_) <- getargs
  testp (case n of
          "1" -> takelist1
