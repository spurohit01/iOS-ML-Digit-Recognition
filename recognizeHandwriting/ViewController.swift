//
//  ViewController.swift
//  recognizeHandwriting
//
//  Created by Sonia Purohit on 6/9/20.
//  Copyright © 2020 Sonia Purohit. All rights reserved.
//

import UIKit
import VideoToolbox

extension UIImage {
    func resizeAndCreatePixelBuffer(alphaInfo: CGImageAlphaInfo) -> CVPixelBuffer? {
        var myPixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, 28, 28,kCVPixelFormatType_OneComponent8, nil, &myPixelBuffer)
        
        guard status == kCVReturnSuccess, let pixelBuffer = myPixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 1))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)
        
        guard let context = CGContext(data: pixelData, width: 28, height: 28, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer), space: CGColorSpaceCreateDeviceGray(), bitmapInfo: alphaInfo.rawValue)
            else {
                return nil
        }
        
        UIGraphicsPushContext(context)
        context.translateBy(x: 0, y: CGFloat(28))
        context.scaleBy(x: 1, y: -1)
        self.draw(in: CGRect(x: 0, y: 0, width: 28, height: 28))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 1))
        return pixelBuffer
    }
}

class ViewController: UIViewController {
    @IBOutlet var drawingView: UIImageView!
    @IBOutlet var predictionDisplay: UILabel!
    @IBOutlet var resizedImageView: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view
        drawingView.backgroundColor = UIColor.black
        drawingView.layer.borderColor = UIColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0).cgColor
        drawingView.layer.masksToBounds = true
        drawingView.contentMode = .scaleToFill
        drawingView.layer.borderWidth = 3
    }
    var lastPoint = CGPoint.zero
    var isSwiping = false
    var color = UIColor.white
    var brushWidth: CGFloat = 8.0
    var opacity: CGFloat = 1.0
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
      guard let touch = touches.first else {
        return
      }
      isSwiping = false
      lastPoint = touch.location(in: drawingView)
    }
    func drawLine(from fromPoint: CGPoint, to toPoint: CGPoint) {
        UIGraphicsBeginImageContextWithOptions(drawingView.frame.size, true, 0.0)
      guard let context = UIGraphicsGetCurrentContext() else {
        return
      }
      drawingView.image?.draw(in: drawingView.bounds)
      context.move(to: fromPoint)
      context.addLine(to: toPoint)
      context.setLineCap(.round)
      context.setBlendMode(.normal)
      context.setLineWidth(brushWidth)
      context.setStrokeColor(color.cgColor)
      context.strokePath()
      drawingView.image = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
      guard let touch = touches.first else {
        return
      }
      isSwiping = true
      let currentPoint = touch.location(in: drawingView)
      drawLine(from: lastPoint, to: currentPoint)
      lastPoint = currentPoint
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
      if !isSwiping {
        drawLine(from: lastPoint, to: lastPoint)
      }
    }
    
    @IBAction func clearDrawing(_ sender: Any) {
        self.drawingView.image = nil
        lastPoint = CGPoint.zero
        self.drawingView.backgroundColor = UIColor.black
        self.resizedImageView.image = nil
        UIGraphicsEndImageContext()
    }
    func resize(image: UIImage, newSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
        image.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage
    }
    @IBAction func recognizePressed(_ sender: Any) {
        var pictureDrawn = self.drawingView.image!
        if let pixelBuffer = pictureDrawn.resizeAndCreatePixelBuffer(alphaInfo: .none) {
            let model = myDigitRecognitionModel()
            guard let prediction = try? model.prediction(conv2d_input: pixelBuffer) else {
                fatalError("Unexpected runtime error.")
            }
            //resizedImageView.image = resizedImage
            var convertedImg: CGImage?
            VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &convertedImg)
            let uiImage = UIImage(cgImage: convertedImg!)
            // and convert that to a SwiftUI image
            resizedImageView.image = uiImage
            let predictedArray = prediction.Identity
            var maxVal = 0.0
            var predictedNumber = -1
            print(predictedArray)
            for index in 0...9{
                var number = Double(predictedArray[index])
                print(number)
                if number > maxVal{
                    predictedNumber = index
                    maxVal = number
                }
            }
            print(predictedNumber)
            predictionDisplay.text = "Digit Recognized is " + String(predictedNumber)
        }
    }
}

