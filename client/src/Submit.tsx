import React, { Component, MouseEvent } from "react";
import { MAX_CHAR_ALLOWED, MIN_CHAR_ALLOWED } from "./config";
import { isRecord } from './record';

type SubmitProps = {
  /** the initial texts shown in the input box */
  initialText: string;

  /** the initial result shown in the output box */
  initialResult: string;

  /** go to the about us page */
  onAboutUsClick: (textSaved: string, resultSaved: string) => void;
}

type SubmitState = {
  /** current text in the input box */
  currentText: string;

  /** current char count of the input text */
  charCount: number;

  /** result of the check from the server */
  currentResult: string

  /** whether we are waiting for the server to respond */
  loading: boolean;

  /** Message telling whether checking was successful or something went wrong */
  message: string
}

export class Submit extends Component<SubmitProps, SubmitState> {

  constructor(props: SubmitProps) {
    super(props);

    this.state = {
      currentText: props.initialText,
      charCount: props.initialText.trim().length,
      currentResult: props.initialResult,
      loading: false,
      message: ""
    }
  }

  render = (): JSX.Element => {
      return <div>
              <div>
                <div>
                  <h1>Input Text to Check</h1>
                  <div style={{ position: 'relative' }}>
                    <input type="text" onChange={this.doTextInputChange} value={this.state.currentText}/>
                    <div style={{ position: 'absolute', bottom: 0, right: 0, margin: 5 }}>
                      {this.state.charCount}
                    </div>
                  </div>
                </div>
                <div>
                  <h1>Checking Result</h1>
                  <p>{this.state.currentResult}</p>
                </div>
              </div>
              <div>
                <button type="button" onClick={this.doSubmitClick}>Check</button>
              </div>
              {this.renderMessage()}
              <div>
                <button type="button" onClick={this.doAboutUsClick}>Back</button>
              </div>
            </div>
  }

  doTextInputChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    this.setState({ currentText: event.target.value, charCount: event.target.value.trim().length, message: "" });
  }
 
  renderMessage = (): JSX.Element => {
    return <p>{this.state.message}</p>
  };

  doSubmitClick = (evt: MouseEvent<HTMLButtonElement>): void => {
    evt.preventDefault();
    const charCount = this.state.currentText.trim().length;
    if (charCount < MIN_CHAR_ALLOWED) {
      this.setState({message: "Please enter at least " + MIN_CHAR_ALLOWED + " characters."});
    } else if (charCount > MAX_CHAR_ALLOWED) {
      this.setState({message: "Please enter no more than " + MAX_CHAR_ALLOWED + " characters."});
    } else {
      fetch("/api/predict", {
        method: "POST", body: JSON.stringify({
          text: this.state.currentText
        }),
        headers: {"Content-Type": "application/json"}})
        .then((res) => this.doSubmitResp(res))
        .catch(() => this.doSubmitError("failed to connect to server"));
    }
  }

  doAboutUsClick = (_evt: MouseEvent<HTMLButtonElement>): void => {
    this.props.onAboutUsClick(this.state.currentText, this.state.currentResult);
  }

  // Called when the server responds to a request for prediction
  doSubmitResp = (res: Response): void => {
    if (res.status === 200) {
      res.json().then(this.doSubmitJson)
          .catch(() => this.doSubmitError("200 response is not JSON"));
    } else if (res.status === 400) {
      res.text().then(this.doSubmitError)
          .catch(() => this.doSubmitError("400 response is not text"));
    } else {
      this.doSubmitError(`bad status code from /api/predict: ${res.status}`);
    }
  }; 

  // Called when the response of prediction is received. 
  doSubmitJson = (data: unknown): void => {
    if (!isRecord(data)) {
      console.error("200 response is not a record", data);
      return;
    }
    if (data.probability === undefined || typeof data.probability !== "number") {
      console.error("200 response missing probability", data);
      return;
    }
    if (data.probability > 1.0 || data.probability < 0.0) {
      console.error("200 response not a valid probability", data);
      return;
    } 
    this.setState({message: ""});
  };

  // Called if an error occurs trying to predict. Display the error message.
  doSubmitError = (msg: string): void => {
    console.error(`Error fetching /api/predict: ${msg}`);
    this.setState({loading: false, message: msg});
  };
}





