import React, { Component, MouseEvent } from "react";
import { LOGO } from "./assets";
import "./styles/AboutUs.css";

type AboutUsProps = {
  /** Most Recent Input Text */
  initialText: string;

  /** Most Recent Result */
  initialResult: string

  /** Goes back to the main submission page */
  onBackClick: (textSaved: string, resultSaved: string) => void;
};

/** Displays the UI of the GPTCheck rsvp application. */
export class AboutUs extends Component<AboutUsProps, {}> {

  constructor(props: AboutUsProps) {
    super(props);

    this.state = {};
  }
  
  render = (): JSX.Element => {
    return <div>
            <img className="logo" src={LOGO} alt="ChatGPT Checker" />
            <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'end', height: '60vh', marginTop: '0' }}>
              <div className="name-card">
                <h2 className="name">Yuekai Xu</h2>
                <p className="info">University of Washington, Seattle</p>
              </div>
              <div className="name-card">
                <h2 className="name">Zhiyuan Jia</h2>
                <p className="info">University of Washington, Seattle</p>
              </div>
            </div>
            <div className="back-container">
              <button type="button" onClick={this.doBackClick}>Back</button>
            </div>
          </div>
  };  

  doBackClick = (_evt: MouseEvent<HTMLButtonElement>): void => {
    this.props.onBackClick(this.props.initialText, this.props.initialResult);
  }
}