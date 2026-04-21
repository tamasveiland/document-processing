targetScope = 'resourceGroup'

@description('Tags that will be applied to all resources')
param tags object = {}

@description('Content Understanding resource name')
param resourceName string

@description('AI Services account name for the project parent')
param aiServicesAccountName string = ''

@description('AI project name for creating the connection')
param aiProjectName string = ''

@description('Id of the user or app to assign application roles')
param principalId string

@description('Principal type of user or app')
param principalType string

@description('Name for the AI Foundry Content Understanding connection')
param connectionName string = 'content-understanding-connection'

@description('Location for all resources')
param location string = resourceGroup().location

// Get reference to the AI Services account and project to access their managed identities
resource aiAccount 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' existing = if (!empty(aiServicesAccountName) && !empty(aiProjectName)) {
  name: aiServicesAccountName

  resource aiProject 'projects' existing = {
    name: aiProjectName
  }
}

// Azure AI Content Understanding resource (dedicated AIServices account)
resource contentUnderstanding 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: resourceName
  location: location
  tags: tags
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: resourceName
    disableLocalAuth: true
    networkAcls: {
      defaultAction: 'Allow'
      virtualNetworkRules: []
      ipRules: []
    }
  }
}

// Role assignment for AI project managed identity to access Content Understanding
resource aiProjectContentUnderstandingRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(aiServicesAccountName) && !empty(aiProjectName)) {
  scope: contentUnderstanding
  name: guid(subscription().id, resourceGroup().id, 'content-understanding-sp-role', aiServicesAccountName, aiProjectName)
  properties: {
    principalId: (aiAccount::aiProject)!.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908') // Cognitive Services User
  }
}

// Role assignment for the deploying user/service principal
resource userContentUnderstandingRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: contentUnderstanding
  name: guid(subscription().id, resourceGroup().id, principalId, 'content-understanding-user-role')
  properties: {
    principalId: principalId
    principalType: principalType
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908') // Cognitive Services User
  }
}

// Create the Content Understanding connection to the AI Foundry project
module contentUnderstandingConnection './connection.bicep' = if (!empty(aiServicesAccountName) && !empty(aiProjectName)) {
  name: 'content-understanding-connection-creation'
  params: {
    aiServicesAccountName: aiServicesAccountName
    aiProjectName: aiProjectName
    connectionConfig: {
      name: connectionName
      category: 'AIServices'
      target: contentUnderstanding.properties.endpoint
      authType: 'AAD'
      isSharedToAll: true
      metadata: {
        ApiType: 'Azure'
        ResourceId: contentUnderstanding.id
        type: 'content_understanding'
      }
    }
    credentials: {}
  }
  dependsOn: [
    aiProjectContentUnderstandingRoleAssignment
  ]
}

output contentUnderstandingName string = contentUnderstanding.name
output contentUnderstandingEndpoint string = contentUnderstanding.properties.endpoint
output contentUnderstandingConnectionName string = contentUnderstandingConnection!.outputs.connectionName
output contentUnderstandingConnectionId string = contentUnderstandingConnection!.outputs.connectionId
output contentUnderstandingResourceId string = contentUnderstanding.id
