//
//  ViewController.swift
//  俺俺Translate
//
//  Created by Koki Tanaka on 2020/02/24.
//  Copyright © 2020 gojiteji.com. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var japanese: UITextField!
    @IBOutlet weak var thai: UILabel!
    var alertController: UIAlertController!
    func alert(title:String, message:String) {
           alertController = UIAlertController(title: title,
                                      message: message,
                                      preferredStyle: .alert)
           alertController.addAction(UIAlertAction(title: "OK",
                                          style: .default,
                                          handler: nil))
           present(alertController, animated: true)
       }
    @IBAction func translate(_ sender: Any) {
        if japanese.text!.count <= 19 {
            thai.text = "翻訳中..."
        
        let urlString = "http://34.83.95.20:80/ja2th"

               let request = NSMutableURLRequest(url: URL(string: urlString)!)

               request.httpMethod = "POST"
               request.addValue("application/json", forHTTPHeaderField: "Content-Type")



               let params:[String:Any] = [
                "ja": japanese.text
               ]

               do{
                   request.httpBody = try JSONSerialization.data(withJSONObject: params, options: .prettyPrinted)

                   let task:URLSessionDataTask = URLSession.shared.dataTask(with: request as URLRequest, completionHandler: {(data,response,error) -> Void in
                       let resultData = String(data: data!, encoding: .utf8)!
                    //print("result:\(resultData)")
                       //print("response:\(response)")
                    self.thai.text=resultData
                   })
                   task.resume()
               }catch{
                   print("Error:\(error)")
                thai.text="ERROR"
                   return
               }
        }else{
            alert(title: "字数が多すぎます",message: "21字数以内に収めてください")
        }
    }
    

    let maxLength: Int = 5

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        japanese.placeholder = "日本語を入力"
        thai.text = ""
            thai.isUserInteractionEnabled = true

            let tg = UITapGestureRecognizer(target: self, action: #selector(ViewController.tappedLabel(_:)))
            thai.addGestureRecognizer(tg)
        }

        @objc func tappedLabel(_ sender:UITapGestureRecognizer) {
            UIPasteboard.general.string = thai.text
            print("clip board :\(UIPasteboard.general.string!)")
    }
        

    func textField(_ textField: UITextField, shouldChangeCharactersIn range: NSRange, replacementString string: String) -> Bool {
        // 入力を反映させたテキストを取得する
        let resultText: String = (textField.text! as NSString).replacingCharacters(in: range, with: string)
        if resultText.count <= 10 {
            return true
        }
        return false
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.view.endEditing(true)
    }
    
    
    

    
}

