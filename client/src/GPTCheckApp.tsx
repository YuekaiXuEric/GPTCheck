import React, { Component } from "react";
import { MAX_CHAR_ALLOWED, MIN_CHAR_ALLOWED } from "./config";
import { Submit } from "./Submit";
import { AboutUs } from "./AboutUs";
import { BACKGROUND } from "./assets";
import "./styles/GPTCheckApp.css";

type Page = {kind: "Submit", initialText: string, initialResult: string} | 
            {kind: "AboutUs", initialText: string, initialResult: string};

type GPTCheckAppState = {
  /** Stores state for the current page of the app to show. */
  show: Page
};

/** Displays the UI of the GPTCheck rsvp application. */
export class GPTCheckApp extends Component<{}, GPTCheckAppState> {

  constructor(props: {}) {
    super(props);

    this.state = {show: {kind: "Submit", initialText: `${MIN_CHAR_ALLOWED} - ${MAX_CHAR_ALLOWED} Characters Allowed`, initialResult: ""}};
  }
  
  render = (): JSX.Element => {
    return (
      <div className="app-container" style={{ backgroundImage: `url(${BACKGROUND})`, backgroundSize: 'cover', backgroundPosition: 'center', height: '100vh', width: '100vw' }}>
        {this.state.show.kind === "Submit" ? (
          <Submit onAboutUsClick={this.doAboutUsClick} initialText={this.state.show.initialText} initialResult={this.state.show.initialResult}/>
        ) : (
          <AboutUs onBackClick={this.doBackClick} initialText={this.state.show.initialText} initialResult={this.state.show.initialResult}/>
        )}
      </div>
    );
  };


  // Switch to the AboutUs Page.
  doAboutUsClick = (textSaved: string, resultSaved: string): void => {
    this.setState({show: {kind: "AboutUs", initialText: textSaved, initialResult: resultSaved}});
  };

  // Come back to the Submit Page.  
  doBackClick = (textSaved: string, resultSaved: string): void => {
    this.setState({show: {kind: "Submit", initialText: textSaved, initialResult: resultSaved}});
  };
}