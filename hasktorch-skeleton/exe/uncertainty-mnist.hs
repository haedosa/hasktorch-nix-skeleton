-- This example is taken from https://penkovsky.com/neural-networks/day8/
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where


import Control.Monad ( forM_, forM, when, (<=<) )
import Control.Monad.Cont ( ContT (..) )
import GHC.Generics
import Pipes hiding ( (~>) )
import qualified Pipes.Prelude as P
import Text.Printf ( printf
                   , PrintfArg )
import Torch
import Torch.Serialize
import Torch.Typed.Vision ( initMnist, MnistData )
import qualified Torch.Vision as V
import Torch.Lens ( HasTypes (..)
                  , over
                  , types )
import Prelude hiding ( exp )

data MLP = MLP
  { fc1 :: Linear,
    fc2 :: Linear,
    fc3 :: Linear
  }
  deriving (Generic, Show, Parameterized)

data MLPSpec = MLPSpec
  { i :: Int,
    h1 :: Int,
    h2 :: Int,
    o :: Int
  }
  deriving (Show, Eq)

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} =
    MLP
      <$> sample (LinearSpec i h1)
      <*> sample (LinearSpec h1 h2)
      <*> sample (LinearSpec h2 o)

(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f

mlp :: MLP -> Bool -> Tensor -> IO Tensor
mlp MLP {..} isStochastic x0 = do
  -- This subnetwork encapsulates the composition
  -- of pure functions
  let sub1 =
          linear fc1
          ~> relu

          ~> linear fc2
          ~> relu

  -- The dropout is applied to the output
  -- of the subnetwork
  x1 <- dropout
          0.1   -- Dropout probability
          isStochastic  -- Activate Dropout when in stochastic mode
          (sub1 x0)  -- Apply dropout to
                     -- the output of `relu` in layer 2

  -- Another linear layer
  let x2 = linear fc3 x1

  -- Finally, logSoftmax, which is numerically more stable
  -- compared to simple log(softmax(x2))
  return $ logSoftmax (Dim 1) x2


toLocalModel :: forall a. HasTypes a Tensor => Device -> DType -> a -> a
toLocalModel device' dtype' = over (types @Tensor @a) (toDevice device')

toLocalModel' :: forall a. HasTypes a Tensor => a -> a
toLocalModel' = toLocalModel (Device CUDA 0) Float

fromLocalModel :: forall a. HasTypes a Tensor => a -> a
fromLocalModel = over (types @Tensor @a) (toDevice (Device CPU 0))


trainLoop
  :: Optimizer o
  => (MLP, o) -> LearningRate -> ListT IO (Tensor, Tensor) -> IO (MLP, o)
trainLoop (model0, opt0) lr = P.foldM step begin done. enumerateData
  where
    isTrain = True
    step :: Optimizer o => (MLP, o) -> ((Tensor, Tensor), Int) -> IO (MLP, o)
    step (model, opt) args = do
      let ((input, label), iter) = toLocalModel' args
      predic <- mlp model isTrain input
      let loss = nllLoss' label predic
      -- Print loss every 100 batches
      when (iter `mod` 100 == 0) $ do
        putStrLn
          $ printf "Batch: %d | Loss: %.2f" iter (asValue loss :: Float)
      runStep model opt loss lr
    done = pure
    begin = pure (model0, opt0)


train :: V.MNIST IO -> Int -> MLP -> IO MLP
train trainMnist epochs net0 = do
    (net', _) <- foldLoop (net0, optimizer) epochs $ \(net', optState) _ ->
      runContT (streamFromMap dsetOpt trainMnist)
      $ trainLoop (net', optState) lr. fst

    return net'
  where
    dsetOpt = datasetOpts workers
    workers = 2
    lr = 1e-4  -- Learning rate
    optimizer = mkAdam 0 beta1 beta2 (flattenParameters net0)
    beta1 = 0.9
    beta2 = 0.999


accuracy :: MLP -> ListT IO (Tensor, Tensor) -> IO Float
accuracy net = P.foldM step begin done. enumerateData
  where
    step :: (Int, Int) -> ((Tensor, Tensor), Int) -> IO (Int, Int)
    step (ac, total) args = do
      let ((input, labels), _) = toLocalModel' args
      -- Compute predictions
      predic <- let stochastic = False
                in argmax (Dim 1) RemoveDim
                     <$> mlp net stochastic input

      let correct = asValue
                        -- Sum those elements
                        $ sumDim (Dim 0) RemoveDim Int64
                        -- Find correct predictions
                        $ predic `eq` labels

      let batchSize = head $ shape predic
      return (ac + correct, total + batchSize)

    -- When done folding, compute the accuracy
    done (ac, total) = pure $ fromIntegral ac / fromIntegral total

    -- Initial errors and totals
    begin = pure (0, 0)

testAccuracy :: V.MNIST IO -> MLP -> IO Float
testAccuracy testStream net = do
    runContT (streamFromMap (datasetOpts 2) testStream) $ accuracy net. fst


save' :: MLP -> FilePath -> IO ()
save' net = save (map toDependent. flattenParameters $ net)

load' :: MLPSpec -> FilePath -> IO MLP
load' spec fpath = do
  params <- mapM makeIndependent <=< load $ fpath
  net0 <- sample spec
  return $ replaceParameters net0 params


predictiveEntropy :: Tensor -> Float
predictiveEntropy predictions =
  let epsilon = 1e-45
      a = meanDim (Dim 0) RemoveDim Float predictions
      b = Torch.log $ a + epsilon
  in asValue $ negate $ sumAll $ a * b


-- Barchart inspired by https://github.com/morishin/ascii-horizontal-barchart/blob/master/src/chart.js
bar :: Floating a => RealFrac a => PrintfArg a => [String] -> [a] -> IO ()
bar lab xs = forM_ ys putStrLn
  where
    ys = let lab' = map (appendSpaces maxLen. Prelude.take maxLabelLen) lab
         in zipWith3 (printf "%s %s %.2f") lab' (showBar xs) xs
    appendSpaces maxN s = let l = length s
                          in s ++ replicate (maxN - l) ' '
    maxLen = Prelude.min maxLabelLen $ _findmax. map length $ lab
    maxLabelLen = 15

showBar :: Floating a => RealFrac a => [a] -> [String]
showBar xs =
  let maxVal = _findmax xs
      maxBarLen = 50
  in map (drawBar maxBarLen maxVal) xs

-- | Formats a bar string
--
-- >>> drawBar 100 1 100
-- "▉"
-- >>> drawBar 100 1.5 100
-- "▉▋"
-- >>> drawBar 100 2 100
-- "▉▉"

drawBar :: Floating a => RealFrac a => a -> a -> a -> String
drawBar maxBarLen maxValue value = bar1
  where
    barLength = value * maxBarLen / maxValue
    wholeNumberPart = Prelude.floor barLength
    fractionalPart = barLength - fromIntegral wholeNumberPart

    bar0 = replicate wholeNumberPart $ _frac _maxFrac
    bar1 = if fractionalPart > 0
      then bar0 ++ [_frac $ Prelude.floor $ fractionalPart * (_maxFrac + 1)]
      else bar0 ++ ""

    _frac 0 = '▏'
    _frac 1 = '▎'
    _frac 2 = '▍'
    _frac 3 = '▋'
    _frac 4 = '▊'
    _frac _ = '▉'

    _maxFrac = 5


_findmax :: Ord a => [a] -> a
_findmax = foldr1 (\x y -> if x >= y then x else y)

displayImage :: MLP -> (Tensor, Tensor) -> IO ()
displayImage model (testImg, testLabel) = do
  let repeatN = 20
      stochastic = True
  preds <- forM [1..repeatN] $ \_ -> exp  -- logSoftmax -> softmax
                                     <$> mlp model stochastic testImg
  pred0 <- mlp model (not stochastic) testImg
  let entropy = predictiveEntropy $ Torch.cat (Dim 0) preds
  -- Select only images with high entropy
  when (entropy > 0.9) $ do
      V.dispImage testImg
      putStr "Entropy "
      print entropy
      -- exp. logSoftmax = softmax
      bar (map show [0..9]) (asValue $ flattenAll $ exp pred0 :: [Float])
      putStrLn $ "Model        : " ++ (show. argmax (Dim 1) RemoveDim. exp $ pred0)
      putStrLn $ "Ground Truth : " ++ show testLabel


main :: IO ()
main = do
  (trainData, testData) <- initMnist "data"
  let trainMnist = V.MNIST {batchSize = 256, mnistData = trainData}
      spec = MLPSpec 784 300 50 10

  -- A train "loader"
  net0 <- toLocalModel' <$> sample spec

  let epochs = 5
  net' <- train trainMnist epochs net0

  -- Saving the trained model
  save' net' "weights.bin"

  net'' <- load' spec "weights.bin"

  let testMnist = V.MNIST { batchSize = 1000, mnistData = testData }

  ac <- testAccuracy testMnist net''
  putStrLn $ "Accuracy " ++ show ac

  let testMnist' = V.MNIST { batchSize = 1, mnistData = testData }
  -- show test images + labels
  forM_ [0 .. 200] $ displayImage (fromLocalModel net'') <=< getItem testMnist'

  putStrLn "Done"
