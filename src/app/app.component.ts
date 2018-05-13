import { Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  // title = 'app';

  linearModel: tf.Sequential;
  prediction: any;

  ngOnInit() {
    this.train();
  }

  async train(): Promise<any> {
    // Define a model for linear regression.
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare the model for training: Specify the loss and the optimizer.
    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});


    // Training data, completely random stuff
    const xs = tf.tensor1d([2, 3, 4, 5, 6, 8]);
    const ys = tf.tensor1d([20, 30, 40, 50, 60, 80]);


    // Train
    await this.linearModel.fit(xs, ys)

    console.log('model trained!');
  }

  predict(val: number) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any;
    this.prediction = Array.from(output.dataSync())[0]
  }
}
