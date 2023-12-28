*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Browser    headless=${False}
    New Page    localhost:3000
    Click       text=Search