{-# OPTIONS_HADDOCK show-extensions #-}

{-|
Module: Cslib
Copyright: (c) Cezary Stankiewicz 2016
Description: 
License: -
Maintainer: c.stankiewicz@wlv.ac.uk
Stability:  experimental
-}

module Cslib (
  module CslibBasics,
  module CslibCM
) where

import CslibBasics hiding (main)
import CslibCM

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
