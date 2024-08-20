import React, { Component, MouseEvent } from "react";
import { MAX_CHAR_ALLOWED, MIN_CHAR_ALLOWED, AI_PROB_THRESHOLD, MIX_PROB_THRESHOLD } from "./config";
import { isRecord } from './record';

type SubmitProps = {
  initialText: string;
  initialResult: string;
  onAboutUsClick: (textSaved: string, resultSaved: string) => void;
}

type SubmitState = {
  currentText: string;
  charCount: number;
  currentResult: string
  loading: boolean;
  message: string;
  abortController: AbortController | null;
}

export class Submit extends Component<SubmitProps, SubmitState> {

  constructor(props: SubmitProps) {
    super(props);

    this.state = {
      currentText: props.initialText,
      charCount: props.initialText.trim().length,
      currentResult: props.initialResult,
      loading: false,
      message: "",
      abortController: null
    }
  }

  render = (): JSX.Element => {
    return <div>
      <img src="./image/logo.jpg" alt="ChatGPT Checker" />
      <div>
        <div>
          <h1>Input Text to Check</h1>
          <div style={{ position: 'relative' }}>
            <input type="text" onChange={this.doTextInputChange} value={this.state.currentText} />
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
      {this.state.loading && this.renderLoadingModal()}
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

  renderLoadingModal = (): JSX.Element => {
    return (
      <div style={{ position: 'fixed', top: '50%', left: 0, width: '100%', height: '50%', background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
        <div style={{ background: 'white', padding: 20, borderRadius: 5 }}>
          <img src="./image/loading.gif" alt="Loading" />
          <button type="button" onClick={this.doCancelClick}>Cancel Submission</button>
        </div>
      </div>
    );
  }

  doSubmitClick = (evt: MouseEvent<HTMLButtonElement>): void => {
    evt.preventDefault();
    const charCount = this.state.currentText.trim().length;
    if (charCount < MIN_CHAR_ALLOWED) {
      this.setState({ message: "Please enter at least " + MIN_CHAR_ALLOWED + " characters." });
    } else if (charCount > MAX_CHAR_ALLOWED) {
      this.setState({ message: "Please enter no more than " + MAX_CHAR_ALLOWED + " characters." });
    } else {
      const abortController = new AbortController();
      this.setState({ loading: true, abortController });
      fetch("/api/predict", {
        method: "POST", body: JSON.stringify({
          text: this.state.currentText
        }),
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal
      })
        .then((res) => this.doSubmitResp(res))
        .catch((error) => this.doHandleError(error, "failed to connect to server"));
    }
  }

  doCancelClick = (): void => {
    if (this.state.abortController) {
      this.state.abortController.abort();
      this.setState({ loading: false, message: "Request cancelled by user." });
    }
  }

  doAboutUsClick = (_evt: MouseEvent<HTMLButtonElement>): void => {
    this.props.onAboutUsClick(this.state.currentText, this.state.currentResult);
  }

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
    if (data.probability >= AI_PROB_THRESHOLD) {
      this.setState({ currentResult: `This text is likely to be generated by an AI model, with AI probability being ${data.probability.toFixed(2)}.`, 
                  loading: false, message: "", abortController: null });
    } else if (data.probability >= MIX_PROB_THRESHOLD) {
      this.setState({ currentResult: `This text is likely to be generated by a mixture of human and AI text, with AI probability being ${data.probability.toFixed(2)}.`, 
                  loading: false, message: "", abortController: null });
    } else {
      this.setState({ currentResult: `This text is likely to be written by a human, with AI probability being ${data.probability.toFixed(2)}.`, 
                  loading: false, message: "", abortController: null });
    }
  };

  doSubmitError = (msg: string): void => {
    console.error("Error fetching /api/predict", `${msg}`);
    this.setState({ loading: false, message: "Something went wrong with the server. Please refresh the page and try again later.", abortController: null });
  };

  doHandleError = (error: Error, msg: string): void => {
    if (error.name === "AbortError") {       
      this.setState({ loading: false, message: "Request cancelled.", abortController: null });
    } else {   
      this.doSubmitError(msg);
    }
  }
}
