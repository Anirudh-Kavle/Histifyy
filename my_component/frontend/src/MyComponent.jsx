import React from "react"
import HTMLFlipBook from "react-pageflip"
import story from "./story_data.json"

class MyComponent extends React.Component {
  renderPages = () => {
    return story.map((page, index) => (
      <div className="demoPage" key={index}>
        <h2>{page.title}</h2>
        <img src={page.img} alt={`Page ${index + 1}`} />
        <div className="pageText">{page.text}</div>
      </div>
    ))
  }

  render() {
    return (
      <div>
        <HTMLFlipBook className="book" width={600} height={600}>
          {this.renderPages()}
        </HTMLFlipBook>
      </div>
    )
  }
}

export default MyComponent
