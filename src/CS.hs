{-# OPTIONS_HADDOCK show-extensions #-}

{-|
Module: CS
Copyright: (c) Cezary Stankiewicz 2016
Description: This module (2 items)
License: -
Maintainer: c.stankiewicz@wlv.ac.uk
Stability   :  experimental
-}

module CS (
  module BasicsCollection,
  module UseOfCM
) where

import BasicsCollection hiding (main)
import UseOfCM

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
