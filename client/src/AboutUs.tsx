import React, { Component, MouseEvent } from "react";

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
            <div>
              <div >
                <h2>Yuekai Xu</h2>
                <p>University of Washington, Seattle</p>
              </div>
              <div>
                <h2>Zhiyuan Jia</h2>
                <p>University of Washington, Seattle</p>
              </div>
            </div>;
            <div>
            <button type="button" onClick={this.doBackClick}>Back</button>
            </div>
          </div>
  };  

  doBackClick = (_evt: MouseEvent<HTMLButtonElement>): void => {
    this.props.onBackClick(this.props.initialText, this.props.initialResult);
  }
}